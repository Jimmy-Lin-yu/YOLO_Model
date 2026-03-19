namespace HW_204N
{
    partial class _常规控制器上位机
    {
        /// <summary>
        /// 必需的设计器变量。
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// 清理所有正在使用的资源。
        /// </summary>
        /// <param name="disposing">如果应释放托管资源，为 true；否则为 false。</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows 窗体设计器生成的代码

        /// <summary>
        /// 设计器支持所需的方法 - 不要
        /// 使用代码编辑器修改此方法的内容。
        /// </summary>
        private void InitializeComponent()
        {
            this.components = new System.ComponentModel.Container();
            this.label1 = new System.Windows.Forms.Label();
            this.tbLight1 = new System.Windows.Forms.TrackBar();
            this.comBoxPorts = new System.Windows.Forms.ComboBox();
            this.label7 = new System.Windows.Forms.Label();
            this.tbState = new System.Windows.Forms.TextBox();
            this.label12 = new System.Windows.Forms.Label();
            this.numLight1 = new System.Windows.Forms.NumericUpDown();
            this.serialPort1 = new System.IO.Ports.SerialPort(this.components);
            this.timer1 = new System.Windows.Forms.Timer(this.components);
            this.trackBar2 = new System.Windows.Forms.TrackBar();
            this.串口状态 = new System.Windows.Forms.GroupBox();
            this.button5 = new System.Windows.Forms.Button();
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.button6 = new System.Windows.Forms.Button();
            this.button7 = new System.Windows.Forms.Button();
            this.button4 = new System.Windows.Forms.Button();
            this.label13 = new System.Windows.Forms.Label();
            this.label14 = new System.Windows.Forms.Label();
            this.tbLight4 = new System.Windows.Forms.TrackBar();
            this.numLight4 = new System.Windows.Forms.NumericUpDown();
            this.button3 = new System.Windows.Forms.Button();
            this.label10 = new System.Windows.Forms.Label();
            this.label11 = new System.Windows.Forms.Label();
            this.tbLight3 = new System.Windows.Forms.TrackBar();
            this.numLight3 = new System.Windows.Forms.NumericUpDown();
            this.button2 = new System.Windows.Forms.Button();
            this.label8 = new System.Windows.Forms.Label();
            this.label9 = new System.Windows.Forms.Label();
            this.tbLight2 = new System.Windows.Forms.TrackBar();
            this.numLight2 = new System.Windows.Forms.NumericUpDown();
            this.button1 = new System.Windows.Forms.Button();
            this.label6 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.label5 = new System.Windows.Forms.Label();
            this.label4 = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.groupBox2 = new System.Windows.Forms.GroupBox();
            this.button8 = new System.Windows.Forms.Button();
            this.textBox1 = new System.Windows.Forms.TextBox();
            ((System.ComponentModel.ISupportInitialize)(this.tbLight1)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLight1)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.trackBar2)).BeginInit();
            this.串口状态.SuspendLayout();
            this.groupBox1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.tbLight4)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLight4)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.tbLight3)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLight3)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.tbLight2)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLight2)).BeginInit();
            this.groupBox2.SuspendLayout();
            this.SuspendLayout();
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(20, 29);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(53, 12);
            this.label1.TabIndex = 0;
            this.label1.Text = "COM端口:";
            // 
            // tbLight1
            // 
            this.tbLight1.Cursor = System.Windows.Forms.Cursors.Default;
            this.tbLight1.LargeChange = 10;
            this.tbLight1.Location = new System.Drawing.Point(54, 46);
            this.tbLight1.Maximum = 255;
            this.tbLight1.Name = "tbLight1";
            this.tbLight1.Orientation = System.Windows.Forms.Orientation.Vertical;
            this.tbLight1.Size = new System.Drawing.Size(45, 216);
            this.tbLight1.TabIndex = 6;
            this.tbLight1.TickStyle = System.Windows.Forms.TickStyle.None;
            this.tbLight1.ValueChanged += new System.EventHandler(this.tbLight1_ValueChanged);
            // 
            // comBoxPorts
            // 
            this.comBoxPorts.FormattingEnabled = true;
            this.comBoxPorts.Location = new System.Drawing.Point(78, 26);
            this.comBoxPorts.Name = "comBoxPorts";
            this.comBoxPorts.Size = new System.Drawing.Size(102, 20);
            this.comBoxPorts.TabIndex = 7;
            this.comBoxPorts.SelectedValueChanged += new System.EventHandler(this.comBoxPorts_SelectedValueChanged);
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(315, 30);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(59, 12);
            this.label7.TabIndex = 8;
            this.label7.Text = "COM 状态:";
            // 
            // tbState
            // 
            this.tbState.Location = new System.Drawing.Point(389, 26);
            this.tbState.Name = "tbState";
            this.tbState.Size = new System.Drawing.Size(123, 21);
            this.tbState.TabIndex = 9;
            // 
            // label12
            // 
            this.label12.AutoSize = true;
            this.label12.Location = new System.Drawing.Point(48, 31);
            this.label12.Name = "label12";
            this.label12.Size = new System.Drawing.Size(35, 12);
            this.label12.TabIndex = 10;
            this.label12.Text = "通道1";
            // 
            // numLight1
            // 
            this.numLight1.Location = new System.Drawing.Point(32, 268);
            this.numLight1.Maximum = new decimal(new int[] {
            255,
            0,
            0,
            0});
            this.numLight1.Name = "numLight1";
            this.numLight1.Size = new System.Drawing.Size(64, 21);
            this.numLight1.TabIndex = 24;
            this.numLight1.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            this.numLight1.ValueChanged += new System.EventHandler(this.numLight1_ValueChanged);
            // 
            // serialPort1
            // 
            this.serialPort1.DataReceived += new System.IO.Ports.SerialDataReceivedEventHandler(this.serialPort1_DataReceived);
            // 
            // timer1
            // 
            this.timer1.Enabled = true;
            this.timer1.Interval = 1000;
            this.timer1.Tick += new System.EventHandler(this.timer1_Tick);
            // 
            // trackBar2
            // 
            this.trackBar2.LargeChange = 10;
            this.trackBar2.Location = new System.Drawing.Point(143, -40);
            this.trackBar2.Maximum = 1000;
            this.trackBar2.Name = "trackBar2";
            this.trackBar2.Size = new System.Drawing.Size(341, 45);
            this.trackBar2.TabIndex = 32;
            this.trackBar2.TickStyle = System.Windows.Forms.TickStyle.None;
            // 
            // 串口状态
            // 
            this.串口状态.Controls.Add(this.button5);
            this.串口状态.Controls.Add(this.label7);
            this.串口状态.Controls.Add(this.label1);
            this.串口状态.Controls.Add(this.trackBar2);
            this.串口状态.Controls.Add(this.tbState);
            this.串口状态.Controls.Add(this.comBoxPorts);
            this.串口状态.Location = new System.Drawing.Point(3, 3);
            this.串口状态.Name = "串口状态";
            this.串口状态.Size = new System.Drawing.Size(666, 64);
            this.串口状态.TabIndex = 34;
            this.串口状态.TabStop = false;
            this.串口状态.Text = "串口状态";
            // 
            // button5
            // 
            this.button5.Location = new System.Drawing.Point(538, 18);
            this.button5.Name = "button5";
            this.button5.Size = new System.Drawing.Size(70, 34);
            this.button5.TabIndex = 33;
            this.button5.Text = "启动连接";
            this.button5.UseVisualStyleBackColor = true;
            this.button5.Click += new System.EventHandler(this.button5_Click);
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.button6);
            this.groupBox1.Controls.Add(this.button7);
            this.groupBox1.Controls.Add(this.button4);
            this.groupBox1.Controls.Add(this.label13);
            this.groupBox1.Controls.Add(this.label14);
            this.groupBox1.Controls.Add(this.tbLight4);
            this.groupBox1.Controls.Add(this.numLight4);
            this.groupBox1.Controls.Add(this.button3);
            this.groupBox1.Controls.Add(this.label10);
            this.groupBox1.Controls.Add(this.label11);
            this.groupBox1.Controls.Add(this.tbLight3);
            this.groupBox1.Controls.Add(this.numLight3);
            this.groupBox1.Controls.Add(this.button2);
            this.groupBox1.Controls.Add(this.label8);
            this.groupBox1.Controls.Add(this.label9);
            this.groupBox1.Controls.Add(this.tbLight2);
            this.groupBox1.Controls.Add(this.numLight2);
            this.groupBox1.Controls.Add(this.button1);
            this.groupBox1.Controls.Add(this.label6);
            this.groupBox1.Controls.Add(this.label2);
            this.groupBox1.Controls.Add(this.label5);
            this.groupBox1.Controls.Add(this.label4);
            this.groupBox1.Controls.Add(this.label3);
            this.groupBox1.Controls.Add(this.tbLight1);
            this.groupBox1.Controls.Add(this.label12);
            this.groupBox1.Controls.Add(this.numLight1);
            this.groupBox1.Location = new System.Drawing.Point(3, 73);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(666, 383);
            this.groupBox1.TabIndex = 35;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "通道亮度调节";
            // 
            // button6
            // 
            this.button6.Location = new System.Drawing.Point(22, 338);
            this.button6.Name = "button6";
            this.button6.Size = new System.Drawing.Size(71, 33);
            this.button6.TabIndex = 53;
            this.button6.Text = "刷新";
            this.button6.UseVisualStyleBackColor = true;
            this.button6.Click += new System.EventHandler(this.button6_Click);
            // 
            // button7
            // 
            this.button7.Location = new System.Drawing.Point(537, 338);
            this.button7.Name = "button7";
            this.button7.Size = new System.Drawing.Size(71, 33);
            this.button7.TabIndex = 52;
            this.button7.Text = "退出";
            this.button7.UseVisualStyleBackColor = true;
            this.button7.Click += new System.EventHandler(this.button7_Click);
            // 
            // button4
            // 
            this.button4.BackColor = System.Drawing.Color.LimeGreen;
            this.button4.Cursor = System.Windows.Forms.Cursors.Hand;
            this.button4.Location = new System.Drawing.Point(494, 299);
            this.button4.Name = "button4";
            this.button4.Size = new System.Drawing.Size(65, 26);
            this.button4.TabIndex = 50;
            this.button4.Text = "打开";
            this.button4.UseVisualStyleBackColor = false;
            this.button4.Click += new System.EventHandler(this.button4_Click);
            // 
            // label13
            // 
            this.label13.AutoSize = true;
            this.label13.Location = new System.Drawing.Point(494, 51);
            this.label13.Name = "label13";
            this.label13.Size = new System.Drawing.Size(23, 12);
            this.label13.TabIndex = 49;
            this.label13.Text = "255";
            // 
            // label14
            // 
            this.label14.AutoSize = true;
            this.label14.Location = new System.Drawing.Point(498, 242);
            this.label14.Name = "label14";
            this.label14.Size = new System.Drawing.Size(11, 12);
            this.label14.TabIndex = 48;
            this.label14.Text = "0";
            // 
            // tbLight4
            // 
            this.tbLight4.Cursor = System.Windows.Forms.Cursors.Default;
            this.tbLight4.LargeChange = 10;
            this.tbLight4.Location = new System.Drawing.Point(516, 46);
            this.tbLight4.Maximum = 255;
            this.tbLight4.Name = "tbLight4";
            this.tbLight4.Orientation = System.Windows.Forms.Orientation.Vertical;
            this.tbLight4.Size = new System.Drawing.Size(45, 216);
            this.tbLight4.TabIndex = 46;
            this.tbLight4.TickStyle = System.Windows.Forms.TickStyle.None;
            this.tbLight4.ValueChanged += new System.EventHandler(this.tbLight4_ValueChanged);
            // 
            // numLight4
            // 
            this.numLight4.Location = new System.Drawing.Point(494, 268);
            this.numLight4.Maximum = new decimal(new int[] {
            255,
            0,
            0,
            0});
            this.numLight4.Name = "numLight4";
            this.numLight4.Size = new System.Drawing.Size(64, 21);
            this.numLight4.TabIndex = 47;
            this.numLight4.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            this.numLight4.ValueChanged += new System.EventHandler(this.numLight4_ValueChanged);
            // 
            // button3
            // 
            this.button3.BackColor = System.Drawing.Color.LimeGreen;
            this.button3.Cursor = System.Windows.Forms.Cursors.Hand;
            this.button3.Location = new System.Drawing.Point(341, 299);
            this.button3.Name = "button3";
            this.button3.Size = new System.Drawing.Size(65, 26);
            this.button3.TabIndex = 45;
            this.button3.Text = "打开";
            this.button3.UseVisualStyleBackColor = false;
            this.button3.Click += new System.EventHandler(this.button3_Click);
            // 
            // label10
            // 
            this.label10.AutoSize = true;
            this.label10.Location = new System.Drawing.Point(341, 51);
            this.label10.Name = "label10";
            this.label10.Size = new System.Drawing.Size(23, 12);
            this.label10.TabIndex = 44;
            this.label10.Text = "255";
            // 
            // label11
            // 
            this.label11.AutoSize = true;
            this.label11.Location = new System.Drawing.Point(345, 242);
            this.label11.Name = "label11";
            this.label11.Size = new System.Drawing.Size(11, 12);
            this.label11.TabIndex = 43;
            this.label11.Text = "0";
            // 
            // tbLight3
            // 
            this.tbLight3.Cursor = System.Windows.Forms.Cursors.Default;
            this.tbLight3.LargeChange = 10;
            this.tbLight3.Location = new System.Drawing.Point(363, 46);
            this.tbLight3.Maximum = 255;
            this.tbLight3.Name = "tbLight3";
            this.tbLight3.Orientation = System.Windows.Forms.Orientation.Vertical;
            this.tbLight3.Size = new System.Drawing.Size(45, 216);
            this.tbLight3.TabIndex = 41;
            this.tbLight3.TickStyle = System.Windows.Forms.TickStyle.None;
            this.tbLight3.ValueChanged += new System.EventHandler(this.tbLight3_ValueChanged);
            // 
            // numLight3
            // 
            this.numLight3.Location = new System.Drawing.Point(341, 268);
            this.numLight3.Maximum = new decimal(new int[] {
            255,
            0,
            0,
            0});
            this.numLight3.Name = "numLight3";
            this.numLight3.Size = new System.Drawing.Size(64, 21);
            this.numLight3.TabIndex = 42;
            this.numLight3.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            this.numLight3.ValueChanged += new System.EventHandler(this.numLight3_ValueChanged);
            // 
            // button2
            // 
            this.button2.BackColor = System.Drawing.Color.LimeGreen;
            this.button2.Cursor = System.Windows.Forms.Cursors.Hand;
            this.button2.Location = new System.Drawing.Point(180, 298);
            this.button2.Name = "button2";
            this.button2.Size = new System.Drawing.Size(65, 26);
            this.button2.TabIndex = 40;
            this.button2.Text = "打开";
            this.button2.UseVisualStyleBackColor = false;
            this.button2.Click += new System.EventHandler(this.button2_Click);
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.Location = new System.Drawing.Point(180, 50);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(23, 12);
            this.label8.TabIndex = 39;
            this.label8.Text = "255";
            // 
            // label9
            // 
            this.label9.AutoSize = true;
            this.label9.Location = new System.Drawing.Point(184, 241);
            this.label9.Name = "label9";
            this.label9.Size = new System.Drawing.Size(11, 12);
            this.label9.TabIndex = 38;
            this.label9.Text = "0";
            // 
            // tbLight2
            // 
            this.tbLight2.Cursor = System.Windows.Forms.Cursors.Arrow;
            this.tbLight2.LargeChange = 10;
            this.tbLight2.Location = new System.Drawing.Point(202, 45);
            this.tbLight2.Maximum = 255;
            this.tbLight2.Name = "tbLight2";
            this.tbLight2.Orientation = System.Windows.Forms.Orientation.Vertical;
            this.tbLight2.Size = new System.Drawing.Size(45, 216);
            this.tbLight2.TabIndex = 36;
            this.tbLight2.TickStyle = System.Windows.Forms.TickStyle.None;
            this.tbLight2.ValueChanged += new System.EventHandler(this.tbLight2_ValueChanged);
            // 
            // numLight2
            // 
            this.numLight2.Location = new System.Drawing.Point(180, 267);
            this.numLight2.Maximum = new decimal(new int[] {
            255,
            0,
            0,
            0});
            this.numLight2.Name = "numLight2";
            this.numLight2.Size = new System.Drawing.Size(64, 21);
            this.numLight2.TabIndex = 37;
            this.numLight2.TextAlign = System.Windows.Forms.HorizontalAlignment.Center;
            this.numLight2.ValueChanged += new System.EventHandler(this.numLight2_ValueChanged);
            // 
            // button1
            // 
            this.button1.BackColor = System.Drawing.Color.LimeGreen;
            this.button1.Cursor = System.Windows.Forms.Cursors.Hand;
            this.button1.Location = new System.Drawing.Point(32, 299);
            this.button1.Name = "button1";
            this.button1.Size = new System.Drawing.Size(65, 26);
            this.button1.TabIndex = 35;
            this.button1.Text = "打开";
            this.button1.UseVisualStyleBackColor = false;
            this.button1.Click += new System.EventHandler(this.button1_Click);
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(32, 51);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(23, 12);
            this.label6.TabIndex = 34;
            this.label6.Text = "255";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(36, 242);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(11, 12);
            this.label2.TabIndex = 33;
            this.label2.Text = "0";
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(510, 31);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(35, 12);
            this.label5.TabIndex = 27;
            this.label5.Text = "通道4";
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(357, 31);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(35, 12);
            this.label4.TabIndex = 26;
            this.label4.Text = "通道3";
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(196, 30);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(35, 12);
            this.label3.TabIndex = 25;
            this.label3.Text = "通道2";
            // 
            // groupBox2
            // 
            this.groupBox2.Controls.Add(this.button8);
            this.groupBox2.Controls.Add(this.textBox1);
            this.groupBox2.Location = new System.Drawing.Point(3, 462);
            this.groupBox2.Name = "groupBox2";
            this.groupBox2.Size = new System.Drawing.Size(666, 132);
            this.groupBox2.TabIndex = 36;
            this.groupBox2.TabStop = false;
            this.groupBox2.Text = "通讯信息              ";
            // 
            // button8
            // 
            this.button8.BackColor = System.Drawing.Color.Turquoise;
            this.button8.Cursor = System.Windows.Forms.Cursors.Hand;
            this.button8.Location = new System.Drawing.Point(573, 20);
            this.button8.Name = "button8";
            this.button8.Size = new System.Drawing.Size(54, 36);
            this.button8.TabIndex = 37;
            this.button8.Text = "清空";
            this.button8.UseVisualStyleBackColor = false;
            this.button8.Click += new System.EventHandler(this.button8_Click);
            // 
            // textBox1
            // 
            this.textBox1.Location = new System.Drawing.Point(8, 16);
            this.textBox1.Multiline = true;
            this.textBox1.Name = "textBox1";
            this.textBox1.ScrollBars = System.Windows.Forms.ScrollBars.Vertical;
            this.textBox1.Size = new System.Drawing.Size(645, 109);
            this.textBox1.TabIndex = 0;
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 12F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(680, 598);
            this.Controls.Add(this.groupBox2);
            this.Controls.Add(this.groupBox1);
            this.Controls.Add(this.串口状态);
            this.Name = "Form1";
            this.Text = " 4通道光源控制器程序V2.4.1";
            this.FormClosing += new System.Windows.Forms.FormClosingEventHandler(this.Form1_FormClosing);
            this.Load += new System.EventHandler(this.Form1_Load);
            this.SizeChanged += new System.EventHandler(this.Form1_SizeChanged);
            ((System.ComponentModel.ISupportInitialize)(this.tbLight1)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLight1)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.trackBar2)).EndInit();
            this.串口状态.ResumeLayout(false);
            this.串口状态.PerformLayout();
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.tbLight4)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLight4)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.tbLight3)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLight3)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.tbLight2)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.numLight2)).EndInit();
            this.groupBox2.ResumeLayout(false);
            this.groupBox2.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TrackBar tbLight1;
        private System.Windows.Forms.ComboBox comBoxPorts;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.TextBox tbState;
        private System.Windows.Forms.Label label12;
        private System.Windows.Forms.NumericUpDown numLight1;
        private System.IO.Ports.SerialPort serialPort1;
        private System.Windows.Forms.Timer timer1;
        private System.Windows.Forms.TrackBar trackBar2;
        private System.Windows.Forms.GroupBox 串口状态;
        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.Button button5;
        private System.Windows.Forms.Button button4;
        private System.Windows.Forms.Label label13;
        private System.Windows.Forms.Label label14;
        private System.Windows.Forms.TrackBar tbLight4;
        private System.Windows.Forms.NumericUpDown numLight4;
        private System.Windows.Forms.Button button3;
        private System.Windows.Forms.Label label10;
        private System.Windows.Forms.Label label11;
        private System.Windows.Forms.TrackBar tbLight3;
        private System.Windows.Forms.NumericUpDown numLight3;
        private System.Windows.Forms.Button button2;
        private System.Windows.Forms.Label label8;
        private System.Windows.Forms.Label label9;
        private System.Windows.Forms.TrackBar tbLight2;
        private System.Windows.Forms.NumericUpDown numLight2;
        private System.Windows.Forms.Button button1;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Button button6;
        private System.Windows.Forms.Button button7;
        private System.Windows.Forms.GroupBox groupBox2;
        private System.Windows.Forms.TextBox textBox1;
        private System.Windows.Forms.Button button8;
    }
}

