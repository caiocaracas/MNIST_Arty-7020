library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity tb_mnist_accel_top is
end entity tb_mnist_accel_top;

architecture sim of tb_mnist_accel_top is

  constant C_S00_AXI_DATA_WIDTH  : integer := 32;
  constant C_S00_AXI_ADDR_WIDTH  : integer := 4;
  constant C_S00_AXIS_TDATA_WIDTH: integer := 32;
  constant C_M00_AXIS_TDATA_WIDTH: integer := 32;

  constant CLK_PERIOD : time := 10 ns;

  -- AXI-Lite signals
  signal s00_axi_aclk    : std_logic := '0';
  signal s00_axi_aresetn : std_logic := '0';
  signal s00_axi_awaddr  : std_logic_vector(C_S00_AXI_ADDR_WIDTH-1 downto 0) := (others => '0');
  signal s00_axi_awprot  : std_logic_vector(2 downto 0) := (others => '0');
  signal s00_axi_awvalid : std_logic := '0';
  signal s00_axi_awready : std_logic;
  signal s00_axi_wdata   : std_logic_vector(C_S00_AXI_DATA_WIDTH-1 downto 0) := (others => '0');
  signal s00_axi_wstrb   : std_logic_vector((C_S00_AXI_DATA_WIDTH/8)-1 downto 0) := (others => '0');
  signal s00_axi_wvalid  : std_logic := '0';
  signal s00_axi_wready  : std_logic;
  signal s00_axi_bresp   : std_logic_vector(1 downto 0);
  signal s00_axi_bvalid  : std_logic;
  signal s00_axi_bready  : std_logic := '0';
  signal s00_axi_araddr  : std_logic_vector(C_S00_AXI_ADDR_WIDTH-1 downto 0) := (others => '0');
  signal s00_axi_arprot  : std_logic_vector(2 downto 0) := (others => '0');
  signal s00_axi_arvalid : std_logic := '0';
  signal s00_axi_arready : std_logic;
  signal s00_axi_rdata   : std_logic_vector(C_S00_AXI_DATA_WIDTH-1 downto 0);
  signal s00_axi_rresp   : std_logic_vector(1 downto 0);
  signal s00_axi_rvalid  : std_logic;
  signal s00_axi_rready  : std_logic := '0';

  -- AXI-Stream slave (image in)
  signal s00_axis_aclk    : std_logic := '0';
  signal s00_axis_aresetn : std_logic := '0';
  signal s00_axis_tready  : std_logic;
  signal s00_axis_tdata   : std_logic_vector(C_S00_AXIS_TDATA_WIDTH-1 downto 0) := (others => '0');
  signal s00_axis_tstrb   : std_logic_vector((C_S00_AXIS_TDATA_WIDTH/8)-1 downto 0) := (others => '0');
  signal s00_axis_tlast   : std_logic := '0';
  signal s00_axis_tvalid  : std_logic := '0';

  -- AXI-Stream master (logits out)
  signal m00_axis_aclk    : std_logic := '0';
  signal m00_axis_aresetn : std_logic := '0';
  signal m00_axis_tvalid  : std_logic;
  signal m00_axis_tdata   : std_logic_vector(C_M00_AXIS_TDATA_WIDTH-1 downto 0);
  signal m00_axis_tstrb   : std_logic_vector((C_M00_AXIS_TDATA_WIDTH/8)-1 downto 0);
  signal m00_axis_tlast   : std_logic;
  signal m00_axis_tready  : std_logic := '1';

  -- Interrupt AXI-Lite (ignored in this test)
  signal s_axi_intr_aclk    : std_logic := '0';
  signal s_axi_intr_aresetn : std_logic := '0';
  signal s_axi_intr_awaddr  : std_logic_vector(4 downto 0) := (others => '0');
  signal s_axi_intr_awprot  : std_logic_vector(2 downto 0) := (others => '0');
  signal s_axi_intr_awvalid : std_logic := '0';
  signal s_axi_intr_awready : std_logic;
  signal s_axi_intr_wdata   : std_logic_vector(31 downto 0) := (others => '0');
  signal s_axi_intr_wstrb   : std_logic_vector(3 downto 0) := (others => '0');
  signal s_axi_intr_wvalid  : std_logic := '0';
  signal s_axi_intr_wready  : std_logic;
  signal s_axi_intr_bresp   : std_logic_vector(1 downto 0);
  signal s_axi_intr_bvalid  : std_logic;
  signal s_axi_intr_bready  : std_logic := '0';
  signal s_axi_intr_araddr  : std_logic_vector(4 downto 0) := (others => '0');
  signal s_axi_intr_arprot  : std_logic_vector(2 downto 0) := (others => '0');
  signal s_axi_intr_arvalid : std_logic := '0';
  signal s_axi_intr_arready : std_logic;
  signal s_axi_intr_rdata   : std_logic_vector(31 downto 0);
  signal s_axi_intr_rresp   : std_logic_vector(1 downto 0);
  signal s_axi_intr_rvalid  : std_logic;
  signal s_axi_intr_rready  : std_logic := '0';
  signal irq                : std_logic;

  -- AXI-Lite register addresses (word-aligned)
  constant ADDR_CTRL    : std_logic_vector(C_S00_AXI_ADDR_WIDTH-1 downto 0) := x"0";
  constant ADDR_STATUS  : std_logic_vector(C_S00_AXI_ADDR_WIDTH-1 downto 0) := x"4";
  constant ADDR_IMG_LEN : std_logic_vector(C_S00_AXI_ADDR_WIDTH-1 downto 0) := x"8";

begin

  -- Share one clock for all domains
  s00_axi_aclk    <= s00_axis_aclk;
  m00_axis_aclk   <= s00_axis_aclk;
  s_axi_intr_aclk <= s00_axis_aclk;

  -- Share one reset for all domains
  s00_axi_aresetn    <= s00_axis_aresetn;
  m00_axis_aresetn   <= s00_axis_aresetn;
  s_axi_intr_aresetn <= s00_axis_aresetn;

  -- Clock generation
  clk_gen : process
  begin
    while true loop
      s00_axis_aclk <= '0';
      wait for CLK_PERIOD/2;
      s00_axis_aclk <= '1';
      wait for CLK_PERIOD/2;
    end loop;
  end process clk_gen;

  -- Reset generation (active-low)
  rst_gen : process
  begin
    s00_axis_aresetn <= '0';
    wait for 5*CLK_PERIOD;
    s00_axis_aresetn <= '1';
    wait;
  end process rst_gen;

  -- DUT instance
  uut : entity work.MNIST_accel
    generic map (
      C_S00_AXI_DATA_WIDTH    => C_S00_AXI_DATA_WIDTH,
      C_S00_AXI_ADDR_WIDTH    => C_S00_AXI_ADDR_WIDTH,
      C_S00_AXIS_TDATA_WIDTH  => C_S00_AXIS_TDATA_WIDTH,
      C_M00_AXIS_TDATA_WIDTH  => C_M00_AXIS_TDATA_WIDTH,
      C_M00_AXIS_START_COUNT  => 32,
      C_S_AXI_INTR_DATA_WIDTH => 32,
      C_S_AXI_INTR_ADDR_WIDTH => 5,
      C_NUM_OF_INTR           => 1,
      C_INTR_SENSITIVITY      => x"FFFFFFFF",
      C_INTR_ACTIVE_STATE     => x"FFFFFFFF",
      C_IRQ_SENSITIVITY       => 1,
      C_IRQ_ACTIVE_STATE      => 1
    )
    port map (
      -- S00_AXI
      s00_axi_aclk    => s00_axi_aclk,
      s00_axi_aresetn => s00_axi_aresetn,
      s00_axi_awaddr  => s00_axi_awaddr,
      s00_axi_awprot  => s00_axi_awprot,
      s00_axi_awvalid => s00_axi_awvalid,
      s00_axi_awready => s00_axi_awready,
      s00_axi_wdata   => s00_axi_wdata,
      s00_axi_wstrb   => s00_axi_wstrb,
      s00_axi_wvalid  => s00_axi_wvalid,
      s00_axi_wready  => s00_axi_wready,
      s00_axi_bresp   => s00_axi_bresp,
      s00_axi_bvalid  => s00_axi_bvalid,
      s00_axi_bready  => s00_axi_bready,
      s00_axi_araddr  => s00_axi_araddr,
      s00_axi_arprot  => s00_axi_arprot,
      s00_axi_arvalid => s00_axi_arvalid,
      s00_axi_arready => s00_axi_arready,
      s00_axi_rdata   => s00_axi_rdata,
      s00_axi_rresp   => s00_axi_rresp,
      s00_axi_rvalid  => s00_axi_rvalid,
      s00_axi_rready  => s00_axi_rready,

      -- S00_AXIS
      s00_axis_aclk    => s00_axis_aclk,
      s00_axis_aresetn => s00_axis_aresetn,
      s00_axis_tready  => s00_axis_tready,
      s00_axis_tdata   => s00_axis_tdata,
      s00_axis_tstrb   => s00_axis_tstrb,
      s00_axis_tlast   => s00_axis_tlast,
      s00_axis_tvalid  => s00_axis_tvalid,

      -- M00_AXIS
      m00_axis_aclk    => m00_axis_aclk,
      m00_axis_aresetn => m00_axis_aresetn,
      m00_axis_tvalid  => m00_axis_tvalid,
      m00_axis_tdata   => m00_axis_tdata,
      m00_axis_tstrb   => m00_axis_tstrb,
      m00_axis_tlast   => m00_axis_tlast,
      m00_axis_tready  => m00_axis_tready,

      -- S_AXI_INTR
      s_axi_intr_aclk    => s_axi_intr_aclk,
      s_axi_intr_aresetn => s_axi_intr_aresetn,
      s_axi_intr_awaddr  => s_axi_intr_awaddr,
      s_axi_intr_awprot  => s_axi_intr_awprot,
      s_axi_intr_awvalid => s_axi_intr_awvalid,
      s_axi_intr_awready => s_axi_intr_awready,
      s_axi_intr_wdata   => s_axi_intr_wdata,
      s_axi_intr_wstrb   => s_axi_intr_wstrb,
      s_axi_intr_wvalid  => s_axi_intr_wvalid,
      s_axi_intr_wready  => s_axi_intr_wready,
      s_axi_intr_bresp   => s_axi_intr_bresp,
      s_axi_intr_bvalid  => s_axi_intr_bvalid,
      s_axi_intr_bready  => s_axi_intr_bready,
      s_axi_intr_araddr  => s_axi_intr_araddr,
      s_axi_intr_arprot  => s_axi_intr_arprot,
      s_axi_intr_arvalid => s_axi_intr_arvalid,
      s_axi_intr_arready => s_axi_intr_arready,
      s_axi_intr_rdata   => s_axi_intr_rdata,
      s_axi_intr_rresp   => s_axi_intr_rresp,
      s_axi_intr_rvalid  => s_axi_intr_rvalid,
      s_axi_intr_rready  => s_axi_intr_rready,
      irq                => irq
    );

  -- AXI-Lite master BFM
  axi_lite_master : process

    procedure axi_write(
      constant addr : in std_logic_vector;
      constant data : in std_logic_vector
    ) is
    begin
      s00_axi_awaddr  <= addr;
      s00_axi_wdata   <= data;
      s00_axi_wstrb   <= (others => '1');
      s00_axi_awvalid <= '1';
      s00_axi_wvalid  <= '1';
      s00_axi_bready  <= '1';

      wait until rising_edge(s00_axi_aclk);
      while (s00_axi_awready = '0' or s00_axi_wready = '0') loop
        wait until rising_edge(s00_axi_aclk);
      end loop;

      s00_axi_awvalid <= '0';
      s00_axi_wvalid  <= '0';

      while s00_axi_bvalid = '0' loop
        wait until rising_edge(s00_axi_aclk);
      end loop;

      wait until rising_edge(s00_axi_aclk);
      s00_axi_bready <= '0';
    end procedure axi_write;

    procedure axi_read(
      constant addr : in  std_logic_vector;
      variable data : out std_logic_vector
    ) is
    begin
      s00_axi_araddr  <= addr;
      s00_axi_arvalid <= '1';
      s00_axi_rready  <= '1';

      wait until rising_edge(s00_axi_aclk);
      while s00_axi_arready = '0' loop
        wait until rising_edge(s00_axi_aclk);
      end loop;
      s00_axi_arvalid <= '0';

      while s00_axi_rvalid = '0' loop
        wait until rising_edge(s00_axi_aclk);
      end loop;

      data := s00_axi_rdata;

      wait until rising_edge(s00_axi_aclk);
      s00_axi_rready <= '0';
    end procedure axi_read;

    variable rd_data : std_logic_vector(C_S00_AXI_DATA_WIDTH-1 downto 0);

  begin
    wait until s00_axi_aresetn = '1';
    wait until rising_edge(s00_axi_aclk);

    -- Configure IMG_LENGTH = 784
    axi_write(ADDR_IMG_LEN, std_logic_vector(to_unsigned(784, C_S00_AXI_DATA_WIDTH)));

    wait for 5*CLK_PERIOD;

    -- Write CTRL with START=1 (IRQ disabled)
    axi_write(ADDR_CTRL, std_logic_vector(to_unsigned(1, C_S00_AXI_DATA_WIDTH)));

    -- Wait for computation and output to complete
    wait for 1000*CLK_PERIOD;

    -- Read STATUS
    axi_read(ADDR_STATUS, rd_data);

    -- Expect DONE=1, BUSY=0 (bits [0]=DONE, [1]=BUSY)
    assert rd_data(0) = '1' and rd_data(1) = '0'
      report "Top-level STATUS register did not indicate DONE=1, BUSY=0"
      severity error;

    report "MNIST_accel top-level AXI-Lite status indicates DONE" severity note;

    wait for 20*CLK_PERIOD;
    assert false report "End of simulation" severity failure;
  end process axi_lite_master;

  -- AXI-Stream image source
  axis_source : process
    constant NUM_BEATS      : integer := 196;  -- 784 bytes / 4 bytes per beat
    constant MAX_WAIT_READY : integer := 200;
    variable beat           : integer;
    variable wait_cnt       : integer;
  begin
    wait until s00_axis_aresetn = '1';
    wait until rising_edge(s00_axis_aclk);
    wait for 20*CLK_PERIOD;

    for beat in 0 to NUM_BEATS-1 loop
      s00_axis_tdata <= std_logic_vector(to_unsigned(beat, C_S00_AXIS_TDATA_WIDTH));
      s00_axis_tstrb <= (s00_axis_tstrb'range => '1');
      if beat = NUM_BEATS-1 then
        s00_axis_tlast <= '1';
      else
        s00_axis_tlast <= '0';
      end if;
      s00_axis_tvalid <= '1';

      wait_cnt := 0;
      while s00_axis_tready = '0' loop
        wait until rising_edge(s00_axis_aclk);
        wait_cnt := wait_cnt + 1;
        if wait_cnt = MAX_WAIT_READY then
          assert false report "s00_axis_tready did not go high during frame" severity error;
          exit;
        end if;
      end loop;

      wait until rising_edge(s00_axis_aclk);

      s00_axis_tvalid <= '0';
      s00_axis_tlast  <= '0';
      s00_axis_tstrb  <= (others => '0');

      wait until rising_edge(s00_axis_aclk);
    end loop;

    s00_axis_tvalid <= '0';
    s00_axis_tlast  <= '0';
    s00_axis_tstrb  <= (others => '0');
    s00_axis_tdata  <= (others => '0');

    wait;
  end process axis_source;

  -- AXI-Stream logits sink
  axis_sink : process
    variable beat_count : integer;
  begin
    beat_count := 0;
    wait until m00_axis_aresetn = '1';
    wait until rising_edge(m00_axis_aclk);

    while true loop
      wait until rising_edge(m00_axis_aclk);
      if m00_axis_tvalid = '1' and m00_axis_tready = '1' then
        beat_count := beat_count + 1;
        if beat_count = 3 then
          assert m00_axis_tlast = '1'
            report "m00_axis_tlast was not asserted on third beat" severity error;
        end if;
      end if;
    end loop;
  end process axis_sink;

end architecture sim;
