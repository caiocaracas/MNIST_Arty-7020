library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity tb_mnist_accel_axi_lite is
end entity tb_mnist_accel_axi_lite;

architecture sim of tb_mnist_accel_axi_lite is

  constant C_S_AXI_DATA_WIDTH : integer := 32;
  constant C_S_AXI_ADDR_WIDTH : integer := 4;

  -- Clock period
  constant CLK_PERIOD : time := 10 ns;

  -- DUT ports
  signal ctrl_start_pulse : std_logic;
  signal ctrl_irq_en      : std_logic;
  signal status_busy      : std_logic := '0';
  signal status_done      : std_logic := '0';
  signal status_error     : std_logic := '0';
  signal img_length       : std_logic_vector(C_S_AXI_DATA_WIDTH - 1 downto 0);

  signal s_axi_aclk    : std_logic                                         := '0';
  signal s_axi_aresetn : std_logic                                         := '0';
  signal s_axi_awaddr  : std_logic_vector(C_S_AXI_ADDR_WIDTH - 1 downto 0) := (others => '0');
  signal s_axi_awprot  : std_logic_vector(2 downto 0)                      := (others => '0');
  signal s_axi_awvalid : std_logic                                         := '0';
  signal s_axi_awready : std_logic;
  signal s_axi_wdata   : std_logic_vector(C_S_AXI_DATA_WIDTH - 1 downto 0)     := (others => '0');
  signal s_axi_wstrb   : std_logic_vector((C_S_AXI_DATA_WIDTH/8) - 1 downto 0) := (others => '0');
  signal s_axi_wvalid  : std_logic                                             := '0';
  signal s_axi_wready  : std_logic;
  signal s_axi_bresp   : std_logic_vector(1 downto 0);
  signal s_axi_bvalid  : std_logic;
  signal s_axi_bready  : std_logic                                         := '0';
  signal s_axi_araddr  : std_logic_vector(C_S_AXI_ADDR_WIDTH - 1 downto 0) := (others => '0');
  signal s_axi_arprot  : std_logic_vector(2 downto 0)                      := (others => '0');
  signal s_axi_arvalid : std_logic                                         := '0';
  signal s_axi_arready : std_logic;
  signal s_axi_rdata   : std_logic_vector(C_S_AXI_DATA_WIDTH - 1 downto 0);
  signal s_axi_rresp   : std_logic_vector(1 downto 0);
  signal s_axi_rvalid  : std_logic;
  signal s_axi_rready  : std_logic := '0';

  -- AXI register addresses (word-aligned)
  -- ADDR_LSB = 2 => word addressing: 0x00, 0x04, 0x08...
  constant ADDR_CTRL    : std_logic_vector(C_S_AXI_ADDR_WIDTH - 1 downto 0) := x"0"; -- 0x00
  constant ADDR_STATUS  : std_logic_vector(C_S_AXI_ADDR_WIDTH - 1 downto 0) := x"4"; -- 0x04
  constant ADDR_IMG_LEN : std_logic_vector(C_S_AXI_ADDR_WIDTH - 1 downto 0) := x"8"; -- 0x08

begin

  -- DUT instance
  dut : entity work.MNIST_accel_slave_lite_v1_0_S00_AXI
    generic map(
      C_S_AXI_DATA_WIDTH => C_S_AXI_DATA_WIDTH,
      C_S_AXI_ADDR_WIDTH => C_S_AXI_ADDR_WIDTH
    )
    port map
    (
      ctrl_start_pulse => ctrl_start_pulse,
      ctrl_irq_en      => ctrl_irq_en,
      status_busy      => status_busy,
      status_done      => status_done,
      status_error     => status_error,
      img_length       => img_length,

      S_AXI_ACLK    => s_axi_aclk,
      S_AXI_ARESETN => s_axi_aresetn,
      S_AXI_AWADDR  => s_axi_awaddr,
      S_AXI_AWPROT  => s_axi_awprot,
      S_AXI_AWVALID => s_axi_awvalid,
      S_AXI_AWREADY => s_axi_awready,
      S_AXI_WDATA   => s_axi_wdata,
      S_AXI_WSTRB   => s_axi_wstrb,
      S_AXI_WVALID  => s_axi_wvalid,
      S_AXI_WREADY  => s_axi_wready,
      S_AXI_BRESP   => s_axi_bresp,
      S_AXI_BVALID  => s_axi_bvalid,
      S_AXI_BREADY  => s_axi_bready,
      S_AXI_ARADDR  => s_axi_araddr,
      S_AXI_ARPROT  => s_axi_arprot,
      S_AXI_ARVALID => s_axi_arvalid,
      S_AXI_ARREADY => s_axi_arready,
      S_AXI_RDATA   => s_axi_rdata,
      S_AXI_RRESP   => s_axi_rresp,
      S_AXI_RVALID  => s_axi_rvalid,
      S_AXI_RREADY  => s_axi_rready
    );

  -- Clock generation
  clk_gen : process
  begin
    while true loop
      s_axi_aclk <= '0';
      wait for CLK_PERIOD/2;
      s_axi_aclk <= '1';
      wait for CLK_PERIOD/2;
    end loop;
  end process clk_gen;

  -- Reset generation (active-low)
  rst_gen : process
  begin
    s_axi_aresetn <= '0';
    wait for 5 * CLK_PERIOD;
    s_axi_aresetn <= '1';
    wait;
  end process rst_gen;

  -- Simple AXI4-Lite BFM: write/read procedures + stimulus
  axi_bfm : process

    -- AXI4-Lite write transaction
    procedure axi_write_reg(
      constant addr : in std_logic_vector;
      constant data : in std_logic_vector
    ) is
    begin
      -- Setup address and data
      s_axi_awaddr  <= addr;
      s_axi_wdata   <= data;
      s_axi_wstrb   <= (others => '1');
      s_axi_awvalid <= '1';
      s_axi_wvalid  <= '1';
      s_axi_bready  <= '1';

      -- Wait for address and data handshake
      wait until rising_edge(s_axi_aclk);
      while (s_axi_awready = '0' or s_axi_wready = '0') loop
        wait until rising_edge(s_axi_aclk);
      end loop;

      -- Deassert address and data valid
      s_axi_awvalid <= '0';
      s_axi_wvalid  <= '0';

      -- Wait for write response
      while s_axi_bvalid = '0' loop
        wait until rising_edge(s_axi_aclk);
      end loop;

      -- One cycle with BVALID and BREADY both high
      wait until rising_edge(s_axi_aclk);
      s_axi_bready <= '0';
    end procedure axi_write_reg;

    -- AXI4-Lite read transaction
    procedure axi_read_reg(
      constant addr : in std_logic_vector;
      variable data : out std_logic_vector
    ) is
    begin
      s_axi_araddr  <= addr;
      s_axi_arvalid <= '1';
      s_axi_rready  <= '1';

      -- Wait for address handshake
      wait until rising_edge(s_axi_aclk);
      while s_axi_arready = '0' loop
        wait until rising_edge(s_axi_aclk);
      end loop;
      s_axi_arvalid <= '0';

      -- Wait for read data
      while s_axi_rvalid = '0' loop
        wait until rising_edge(s_axi_aclk);
      end loop;

      data := s_axi_rdata;

      -- One cycle with RVALID and RREADY both high
      wait until rising_edge(s_axi_aclk);
      s_axi_rready <= '0';
    end procedure axi_read_reg;

    -- Readback buffer
    variable rd_data : std_logic_vector(C_S_AXI_DATA_WIDTH - 1 downto 0);

  begin
    -- Wait for reset deassertion
    wait until s_axi_aresetn = '1';
    wait until rising_edge(s_axi_aclk);

    -- 1) Write IMG_LENGTH = 784 (0x00000310) and check output
    axi_write_reg(
    ADDR_IMG_LEN,
    std_logic_vector(to_unsigned(784, C_S_AXI_DATA_WIDTH))
    );
    -- Allow some cycles for register to update
    wait for 3 * CLK_PERIOD;

    assert img_length = std_logic_vector(to_unsigned(784, C_S_AXI_DATA_WIDTH))
    report "IMG_LENGTH register did not latch expected value 784"
      severity error;

    -- 2) Write CTRL: set START=1, IRQ_EN=1 and check IRQ_EN latch
    -- First clear CTRL
    axi_write_reg(
    ADDR_CTRL,
    (C_S_AXI_DATA_WIDTH - 1 downto 0 => '0')
    );

    -- Now write START=1 (bit 0), IRQ_EN=1 (bit 1) => value = 3
    axi_write_reg(
    ADDR_CTRL,
    std_logic_vector(to_unsigned(3, C_S_AXI_DATA_WIDTH))
    );

    -- Wait a few cycles
    wait for 3 * CLK_PERIOD;

    -- Check IRQ_EN latched
    assert ctrl_irq_en = '1'
    report "ctrl_irq_en did not latch to '1' after CTRL write with IRQ_EN=1"
      severity error;

    -- 3) Exercise STATUS: busy/done/error -> readback at 0x04
    -- Scenario A: busy = 1, done = 0, error = 0
    status_busy  <= '1';
    status_done  <= '0';
    status_error <= '0';
    wait for 2 * CLK_PERIOD;

    axi_read_reg(ADDR_STATUS, rd_data);
    assert rd_data(1) = '1' and rd_data(0) = '0' and rd_data(2) = '0'
    report "STATUS readback mismatch for busy=1, done=0, error=0"
      severity error;

    -- Scenario B: busy = 0, done = 1, error = 1
    status_busy  <= '0';
    status_done  <= '1';
    status_error <= '1';
    wait for 2 * CLK_PERIOD;

    axi_read_reg(ADDR_STATUS, rd_data);
    assert rd_data(0) = '1' and rd_data(1) = '0' and rd_data(2) = '1'
    report "STATUS readback mismatch for busy=0, done=1, error=1"
      severity error;

    -- End of simulation
    report "AXI-Lite control interface test completed successfully" severity note;
    wait for 10 * CLK_PERIOD;
    assert false report "End of simulation" severity failure;
  end process axi_bfm;

end architecture sim;
