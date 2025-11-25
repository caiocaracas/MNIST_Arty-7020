library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity tb_mnist_accel_master_stream is
end entity tb_mnist_accel_master_stream;

architecture sim of tb_mnist_accel_master_stream is

  constant C_M_AXIS_TDATA_WIDTH : integer := 32;
  constant CLK_PERIOD           : time    := 10 ns;

  signal M_AXIS_ACLK    : std_logic := '0';
  signal M_AXIS_ARESETN : std_logic := '0';
  signal M_AXIS_TVALID  : std_logic;
  signal M_AXIS_TDATA   : std_logic_vector(C_M_AXIS_TDATA_WIDTH-1 downto 0);
  signal M_AXIS_TSTRB   : std_logic_vector((C_M_AXIS_TDATA_WIDTH/8)-1 downto 0);
  signal M_AXIS_TLAST   : std_logic;
  signal M_AXIS_TREADY  : std_logic := '1';

  signal logits_data    : std_logic_vector(8*10-1 downto 0) := (others => '0');
  signal logits_valid   : std_logic := '0';
  signal logits_sent    : std_logic;

begin

  -- DUT instance
  dut : entity work.MNIST_accel_master_stream_v1_0_M00_AXIS
    generic map (
      C_M_AXIS_TDATA_WIDTH => C_M_AXIS_TDATA_WIDTH,
      C_M_AXIS_START_COUNT => 32
    )
    port map (
      M_AXIS_ACLK    => M_AXIS_ACLK,
      M_AXIS_ARESETN => M_AXIS_ARESETN,
      M_AXIS_TVALID  => M_AXIS_TVALID,
      M_AXIS_TDATA   => M_AXIS_TDATA,
      M_AXIS_TSTRB   => M_AXIS_TSTRB,
      M_AXIS_TLAST   => M_AXIS_TLAST,
      M_AXIS_TREADY  => M_AXIS_TREADY,
      logits_data    => logits_data,
      logits_valid   => logits_valid,
      logits_sent    => logits_sent
    );

  -- Clock generation
  clk_gen : process
  begin
    while true loop
      M_AXIS_ACLK <= '0';
      wait for CLK_PERIOD/2;
      M_AXIS_ACLK <= '1';
      wait for CLK_PERIOD/2;
    end loop;
  end process clk_gen;

  -- Reset generation (active-low)
  rst_gen : process
  begin
    M_AXIS_ARESETN <= '0';
    wait for 5*CLK_PERIOD;
    M_AXIS_ARESETN <= '1';
    wait;
  end process rst_gen;

  -- Stimulus and checks
  stim_proc : process
    constant NUM_BEATS      : integer := 3;
    constant MAX_WAIT_BEAT  : integer := 200;

    constant LOGITS_VEC : std_logic_vector(8*10-1 downto 0) :=
      x"09080706050403020100";

    type data_array_t is array (0 to NUM_BEATS-1) of std_logic_vector(31 downto 0);
    type strb_array_t is array (0 to NUM_BEATS-1) of std_logic_vector(3 downto 0);
    type last_array_t is array (0 to NUM_BEATS-1) of std_logic;

    constant EXP_DATA : data_array_t := (
      0 => x"03020100",
      1 => x"07060504",
      2 => x"00000908"
    );

    constant EXP_STRB : strb_array_t := (
      0 => "1111",
      1 => "1111",
      2 => "0011"
    );

    constant EXP_LAST : last_array_t := (
      0 => '0',
      1 => '0',
      2 => '1'
    );

    variable beat           : integer;
    variable wait_cnt       : integer;
    variable logits_sent_seen : boolean := false;

  begin
    -- Wait for reset deassertion
    wait until M_AXIS_ARESETN = '1';
    wait until rising_edge(M_AXIS_ACLK);

    -- Drive logits_data = [0..9]
    logits_data <= LOGITS_VEC;

    -- Small delay in IDLE
    wait for 5*CLK_PERIOD;

    -- Pulse logits_valid for one cycle
    logits_valid <= '1';
    wait until rising_edge(M_AXIS_ACLK);
    logits_valid <= '0';

    -- Receive NUM_BEATS AXIS beats
    for beat in 0 to NUM_BEATS-1 loop
      wait_cnt := 0;

      -- Wait for TVALID and TREADY with timeout
      while not (M_AXIS_TVALID = '1' and M_AXIS_TREADY = '1') loop
        wait until rising_edge(M_AXIS_ACLK);
        wait_cnt := wait_cnt + 1;
        if wait_cnt = MAX_WAIT_BEAT then
          assert false report "Timeout waiting for AXIS beat from master" severity error;
          exit;
        end if;
      end loop;

      -- Handshake happened in this cycle, check outputs
      assert M_AXIS_TDATA = EXP_DATA(beat)
        report "TDATA mismatch on beat " & integer'image(beat)
        severity error;

      assert M_AXIS_TSTRB = EXP_STRB(beat)
        report "TSTRB mismatch on beat " & integer'image(beat)
        severity error;

      assert M_AXIS_TLAST = EXP_LAST(beat)
        report "TLAST mismatch on beat " & integer'image(beat)
        severity error;

      -- Observe logits_sent after the last beat
      if beat = NUM_BEATS-1 then
        wait until rising_edge(M_AXIS_ACLK);
        if logits_sent = '1' then
          logits_sent_seen := true;
        end if;
      else
        wait until rising_edge(M_AXIS_ACLK);
      end if;
    end loop;

    -- After all beats, TVALID should eventually go low
    wait_cnt := 0;
    while M_AXIS_TVALID = '1' loop
      wait until rising_edge(M_AXIS_ACLK);
      wait_cnt := wait_cnt + 1;
      if wait_cnt = MAX_WAIT_BEAT then
        assert false report "M_AXIS_TVALID did not deassert after last beat" severity error;
        exit;
      end if;
    end loop;

    -- Check logits_sent pulse was seen
    assert logits_sent_seen
      report "logits_sent was not asserted after last beat" severity error;

    report "AXI-Stream master interface test completed successfully" severity note;
    wait for 10*CLK_PERIOD;
    assert false report "End of simulation" severity failure;
  end process stim_proc;

end architecture sim;
