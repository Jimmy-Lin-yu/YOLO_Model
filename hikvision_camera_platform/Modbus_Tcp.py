#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import time
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from pymodbus.client import ModbusTcpClient


# 讓你可以寫：arg.send(OK) / arg.send(NG)
OK = "OK"
NG = "NG"


def _id_kwargs(fn: Callable[..., Any], device_id: int) -> Dict[str, Any]:
    """
    不同 pymodbus 版本，站號參數名可能不同：
      - device_id=
      - slave=
      - unit=
    若該版本都不支援，就回 {}（不帶站號參數）。
    """
    try:
        params = inspect.signature(fn).parameters
    except Exception:
        return {}

    if "device_id" in params:
        return {"device_id": device_id}
    if "slave" in params:
        return {"slave": device_id}
    if "unit" in params:
        return {"unit": device_id}
    return {}


def _read_hr(client: ModbusTcpClient, addr: int, device_id: int) -> int:
    """兼容不同 pymodbus 版本 read_holding_registers 的參數寫法。"""
    kw = _id_kwargs(client.read_holding_registers, device_id)
    try:
        rr = client.read_holding_registers(addr, count=1, **kw)
    except TypeError:
        # 有些版本 count 不是 keyword-only
        rr = client.read_holding_registers(addr, 1, **kw)

    if hasattr(rr, "isError") and rr.isError():
        raise RuntimeError(f"read_holding_registers error: {rr}")
    return int(rr.registers[0])


def _write_hr(client: ModbusTcpClient, addr: int, value: int, device_id: int) -> None:
    """兼容不同 pymodbus 版本 write_register 的站號參數。"""
    kw = _id_kwargs(client.write_register, device_id)
    rr = client.write_register(addr, value, **kw)
    if hasattr(rr, "isError") and rr.isError():
        raise RuntimeError(f"write_register error: {rr}")


def _read_coil(client: ModbusTcpClient, addr: int, device_id: int) -> int:
    """讀 Coil：回傳 0/1（相容不同 pymodbus 版本參數寫法）。"""
    kw = _id_kwargs(client.read_coils, device_id)
    try:
        rr = client.read_coils(addr, count=1, **kw)
    except TypeError:
        rr = client.read_coils(addr, 1, **kw)

    if hasattr(rr, "isError") and rr.isError():
        raise RuntimeError(f"read_coils error: {rr}")

    # rr.bits 是 list[bool]
    return 1 if rr.bits and rr.bits[0] else 0


def _write_coil(client: ModbusTcpClient, addr: int, value: bool | int, device_id: int) -> None:
    """寫 Coil：value 可傳 True/False 或 1/0（相容不同 pymodbus 版本站號參數）。"""
    val = bool(value)
    kw = _id_kwargs(client.write_coil, device_id)

    rr = client.write_coil(addr, val, **kw)
    if hasattr(rr, "isError") and rr.isError():
        raise RuntimeError(f"write_coil error: {rr}")






@dataclass
class AIPLCModbus:
    host: str
    port: int = 502
    device_id: int = 1          # unit/slave id
    address_base: int = 0       # 0-based=0；若你發現要減一才對，改成 1
    tx_addr: int = 1000         # PC->PLC (D100) 項目：完成接收
    rx_addr: int = 1002        # PLC->PC (D102) echo 項目：完成回傳
    ok_value: int = 1
    ng_value: int = 0
    timeout: float = 2.0
    poll_interval: float = 0.05

    def _addr(self, a: int) -> int:
        return a - self.address_base

    def send(self, what: str, wait_echo: bool = True, wait_timeout: float = 5.0) -> Optional[int]:
        """
        what: OK / NG
        wait_echo: 是否等待 PLC ACK
        回傳：PLC 的 rx 值（0/1）
        """
        what_u = what.strip().upper()
        if what_u not in ("OK", "NG"):
            raise ValueError("what must be OK or NG")

        value = self.ok_value if what_u == "OK" else self.ng_value
        tx = self._addr(self.tx_addr)
        rx = self._addr(self.rx_addr)

        client = ModbusTcpClient(self.host, port=self.port, timeout=self.timeout)
        if not client.connect():
            raise ConnectionError(f"connect failed: {self.host}:{self.port}")

        try:
            print(f"[PC] Send {what_u} -> PLC (coil {self.tx_addr})")
            # _write_coil(client, tx, value, self.device_id)
            _write_hr(client, tx, value, self.device_id)
            if not wait_echo:
                return None

            print(f"[PC] Waiting PLC ACK on coil {self.rx_addr} ...")

            t0 = time.time()
            while time.time() - t0 < wait_timeout:
                echo = _read_hr(client, self.rx_addr, self.device_id)
                print(f"[PC] PLC rx = {echo}")

                if echo == 1:
                    print("[PC] PLC ACK received ✅")
                    return echo

                time.sleep(self.poll_interval)

            raise TimeoutError("wait PLC ACK timeout")

        finally:
            client.close()



if __name__ == "__main__":
    # ===== 你只要改這裡就好 =====
    PLC_IP = "192.168.0.222"

    # 建立物件（你也可以改 address_base / device_id / tx_addr / rx_addr）
    arg = AIPLCModbus(
        host=PLC_IP,
        port=502,
        device_id=1,
        address_base=0,   # 若寫 1000 沒反應，改成 1（會自動 -1）
        tx_addr=50,     #  PC -> PLC  tx = transmit（送出）
        rx_addr=53,     #  PLC -> PC  rx = receive（接收）
        ok_value=1,
        ng_value=0
    )

    # 直接改這行：OK / NG
    # arg.send("OK", wait_echo=False)  # <- 改成 arg.send("NG") 就會送 NG(0)

    # 如果你要等 PLC 回寫 1001 做確認：
    ack = arg.send("NG", wait_echo=True, wait_timeout=10.0)
    print("Final PLC ACK =", ack)
