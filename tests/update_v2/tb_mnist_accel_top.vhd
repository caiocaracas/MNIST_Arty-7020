library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.env.all;

-- requires w1.mem..w4.mem and b1.mem..b4.mem in the simulator run directory

entity tb_mnist_accel_top is end;

architecture sim of tb_mnist_accel_top is
  constant CLK_PERIOD:time:=10 ns;
  constant IMG_BYTES:integer:=784;
  constant IMG_WORDS:integer:=IMG_BYTES/4;
  constant AXI_ADDR_W:integer:=4;
  constant AXI_DATA_W:integer:=32;
  signal clk:std_logic:='0';
  signal rstn:std_logic:='0';
  signal awaddr:std_logic_vector(AXI_ADDR_W-1 downto 0):=(others=>'0');
  signal awprot:std_logic_vector(2 downto 0):=(others=>'0');
  signal awvalid:std_logic:='0';
  signal awready:std_logic;
  signal wdata:std_logic_vector(AXI_DATA_W-1 downto 0):=(others=>'0');
  signal wstrb:std_logic_vector(3 downto 0):=(others=>'1');
  signal wvalid:std_logic:='0';
  signal wready:std_logic;
  signal bresp:std_logic_vector(1 downto 0);
  signal bvalid:std_logic;
  signal bready:std_logic:='1';
  signal araddr:std_logic_vector(AXI_ADDR_W-1 downto 0):=(others=>'0');
  signal arprot:std_logic_vector(2 downto 0):=(others=>'0');
  signal arvalid:std_logic:='0';
  signal arready:std_logic;
  signal rdata:std_logic_vector(AXI_DATA_W-1 downto 0);
  signal rresp:std_logic_vector(1 downto 0);
  signal rvalid:std_logic;
  signal rready:std_logic:='1';
  signal s_tready:std_logic;
  signal s_tdata:std_logic_vector(31 downto 0):=(others=>'0');
  signal s_tstrb:std_logic_vector(3 downto 0):=(others=>'1');
  signal s_tlast:std_logic:='0';
  signal s_tvalid:std_logic:='0';
  signal m_tvalid:std_logic;
  signal m_tdata:std_logic_vector(31 downto 0);
  signal m_tstrb:std_logic_vector(3 downto 0);
  signal m_tlast:std_logic;
  signal m_tready:std_logic:='1';
  signal i_awaddr:std_logic_vector(4 downto 0):=(others=>'0');
  signal i_awprot:std_logic_vector(2 downto 0):=(others=>'0');
  signal i_awvalid:std_logic:='0';
  signal i_awready:std_logic;
  signal i_wdata:std_logic_vector(31 downto 0):=(others=>'0');
  signal i_wstrb:std_logic_vector(3 downto 0):=(others=>'0');
  signal i_wvalid:std_logic:='0';
  signal i_wready:std_logic;
  signal i_bresp:std_logic_vector(1 downto 0);
  signal i_bvalid:std_logic;
  signal i_bready:std_logic:='1';
  signal i_araddr:std_logic_vector(4 downto 0):=(others=>'0');
  signal i_arprot:std_logic_vector(2 downto 0):=(others=>'0');
  signal i_arvalid:std_logic:='0';
  signal i_arready:std_logic;
  signal i_rdata:std_logic_vector(31 downto 0);
  signal i_rresp:std_logic_vector(1 downto 0);
  signal i_rvalid:std_logic;
  signal i_rready:std_logic:='1';
  signal irq:std_logic;
begin
  clk<=not clk after CLK_PERIOD/2;
  dut:entity work.MNIST_accel
    port map(
      s00_axi_aclk=>clk,s00_axi_aresetn=>rstn,
      s00_axi_awaddr=>awaddr,s00_axi_awprot=>awprot,s00_axi_awvalid=>awvalid,s00_axi_awready=>awready,
      s00_axi_wdata=>wdata,s00_axi_wstrb=>wstrb,s00_axi_wvalid=>wvalid,s00_axi_wready=>wready,
      s00_axi_bresp=>bresp,s00_axi_bvalid=>bvalid,s00_axi_bready=>bready,
      s00_axi_araddr=>araddr,s00_axi_arprot=>arprot,s00_axi_arvalid=>arvalid,s00_axi_arready=>arready,
      s00_axi_rdata=>rdata,s00_axi_rresp=>rresp,s00_axi_rvalid=>rvalid,s00_axi_rready=>rready,
      s00_axis_aclk=>clk,s00_axis_aresetn=>rstn,
      s00_axis_tready=>s_tready,s00_axis_tdata=>s_tdata,s00_axis_tstrb=>s_tstrb,s00_axis_tlast=>s_tlast,s00_axis_tvalid=>s_tvalid,
      m00_axis_aclk=>clk,m00_axis_aresetn=>rstn,
      m00_axis_tvalid=>m_tvalid,m00_axis_tdata=>m_tdata,m00_axis_tstrb=>m_tstrb,m00_axis_tlast=>m_tlast,m00_axis_tready=>m_tready,
      s_axi_intr_aclk=>clk,s_axi_intr_aresetn=>rstn,
      s_axi_intr_awaddr=>i_awaddr,s_axi_intr_awprot=>i_awprot,s_axi_intr_awvalid=>i_awvalid,s_axi_intr_awready=>i_awready,
      s_axi_intr_wdata=>i_wdata,s_axi_intr_wstrb=>i_wstrb,s_axi_intr_wvalid=>i_wvalid,s_axi_intr_wready=>i_wready,
      s_axi_intr_bresp=>i_bresp,s_axi_intr_bvalid=>i_bvalid,s_axi_intr_bready=>i_bready,
      s_axi_intr_araddr=>i_araddr,s_axi_intr_arprot=>i_arprot,s_axi_intr_arvalid=>i_arvalid,s_axi_intr_arready=>i_arready,
      s_axi_intr_rdata=>i_rdata,s_axi_intr_rresp=>i_rresp,s_axi_intr_rvalid=>i_rvalid,s_axi_intr_rready=>i_rready,
      irq=>irq);
  stim:process
    procedure axi_write(constant a:in std_logic_vector;constant d:in std_logic_vector) is
    begin
      awaddr<=a;wdata<=d;awvalid<='1';wvalid<='1';
      wait until rising_edge(clk) and awready='1' and wready='1';
      awvalid<='0';wvalid<='0';
      wait until rising_edge(clk) and bvalid='1';
      assert bresp="00" report "axi write bresp" severity failure;
      wait until rising_edge(clk);
    end procedure;
    procedure axi_read(constant a:in std_logic_vector;variable d:out std_logic_vector) is
    begin
      araddr<=a;arvalid<='1';
      wait until rising_edge(clk) and arready='1';
      arvalid<='0';
      wait until rising_edge(clk) and rvalid='1';
      assert rresp="00" report "axi read rresp" severity failure;
      d:=rdata;
      wait until rising_edge(clk);
    end procedure;
    procedure axis_send_word(constant data:in std_logic_vector(31 downto 0);constant last:in std_logic) is
    begin
      s_tdata<=data;s_tlast<=last;s_tvalid<='1';
      wait until rising_edge(clk) and s_tready='1';
      s_tvalid<='0';s_tlast<='0';
      wait until rising_edge(clk);
    end procedure;
    procedure axis_recv_logits(variable l0:out integer;variable l1:out integer;variable l2:out integer;variable l3:out integer;variable l4:out integer;variable l5:out integer;variable l6:out integer;variable l7:out integer;variable l8:out integer;variable l9:out integer) is
      variable bytes:std_logic_vector(79 downto 0):=(others=>'0');
      variable v0,v1,v2:std_logic_vector(31 downto 0);
    begin
      while m_tvalid/='1' loop wait until rising_edge(clk);end loop;
      wait until rising_edge(clk) and m_tvalid='1' and m_tready='1';
      v0:=m_tdata;
      assert m_tstrb="1111" report "unexpected tstrb" severity failure;
      assert m_tlast='0' report "unexpected tlast" severity failure;
      wait until rising_edge(clk) and m_tvalid='1' and m_tready='1';
      v1:=m_tdata;
      assert m_tstrb="1111" report "unexpected tstrb" severity failure;
      assert m_tlast='0' report "unexpected tlast" severity failure;
      wait until rising_edge(clk) and m_tvalid='1' and m_tready='1';
      v2:=m_tdata;
      assert m_tstrb(1 downto 0)="11" report "unexpected tstrb last" severity failure;
      assert m_tlast='1' report "missing tlast" severity failure;
      bytes(31 downto 0):=v0;
      bytes(63 downto 32):=v1;
      bytes(79 downto 64):=v2(15 downto 0);
      l0:=to_integer(signed(bytes(7 downto 0)));
      l1:=to_integer(signed(bytes(15 downto 8)));
      l2:=to_integer(signed(bytes(23 downto 16)));
      l3:=to_integer(signed(bytes(31 downto 24)));
      l4:=to_integer(signed(bytes(39 downto 32)));
      l5:=to_integer(signed(bytes(47 downto 40)));
      l6:=to_integer(signed(bytes(55 downto 48)));
      l7:=to_integer(signed(bytes(63 downto 56)));
      l8:=to_integer(signed(bytes(71 downto 64)));
      l9:=to_integer(signed(bytes(79 downto 72)));
      wait until rising_edge(clk);
    end procedure;
    variable st:std_logic_vector(31 downto 0);
    variable l0,l1,l2,l3,l4,l5,l6,l7,l8,l9:integer;
  begin
    rstn<='0';
    wait for 20*CLK_PERIOD;
    rstn<='1';
    wait for 5*CLK_PERIOD;
    axi_write(x"8",std_logic_vector(to_unsigned(IMG_BYTES,32)));
    axi_write(x"0",x"00000001");
    for i in 0 to IMG_WORDS-2 loop axis_send_word(std_logic_vector(to_unsigned(i,32)),'0');end loop;
    axis_send_word(std_logic_vector(to_unsigned(IMG_WORDS-1,32)),'1');
    axis_recv_logits(l0,l1,l2,l3,l4,l5,l6,l7,l8,l9);
    report "logits: "&integer'image(l0)&","&integer'image(l1)&","&integer'image(l2)&","&integer'image(l3)&","&integer'image(l4)&","&integer'image(l5)&","&integer'image(l6)&","&integer'image(l7)&","&integer'image(l8)&","&integer'image(l9);
    axi_read(x"4",st);
    assert st(1)='0' report "busy did not clear" severity failure;
    stop(0);
  end process;
end architecture;
