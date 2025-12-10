from pymodbus.client import ModbusTcpClient
import socket, time

LOCAL_IP = "192.168.250.3"   # 你剛加在電腦上的位址
HMI_IP   = "192.168.250.2"    # HMI
UNIT_ID  = 1                  # 若 HMI 當 gateway，這裡要填 PLC 的 Slave ID；若只是 HMI 內存，通常 1

# 連線前測試 502
socket.create_connection((HMI_IP, 502), timeout=3, source_address=(LOCAL_IP, 0)).close()

client = ModbusTcpClient(HMI_IP, port=502, timeout=3,
                         source_address=(LOCAL_IP, 0))
if not client.connect():
    raise SystemExit("連不上 HMI：沒開 Modbus Server 或被防火牆擋")

def w(addr, val):
    res = client.write_register(address=addr, value=val, slave=UNIT_ID)  # pymodbus 3.x 用 slave=
    if res.isError():
        raise RuntimeError(res)

def r(addr, n=1):
    res = client.read_holding_registers(address=addr, count=n, slave=UNIT_ID)
    if res.isError():
        raise RuntimeError(res)
    return res.registers if n>1 else res.registers[0]

# ★ 下面的位址必須依「HMI 的位址表」來填 ★
# 例：若 HMI 定義 40001=角度(×10)，40002=啟動
w(0, 450)      # 有些設備 40001 要寫 address=0
w(1, 1)        # 啟動
while True:
    done = r(2)  # 假設 40003=完成旗標
    if done == 1:
        break
    time.sleep(0.05)

client.close()
