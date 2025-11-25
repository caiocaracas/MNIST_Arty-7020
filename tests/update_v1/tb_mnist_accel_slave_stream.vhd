library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity tb_mnist_accel_slave_stream is
end entity tb_mnist_accel_slave_stream;

architecture sim of tb_mnist_accel_slave_stream is

  constant C_S_AXIS_TDATA_WIDTH : integer := 32;
  constant CLK_PERIOD           : time    := 10 ns;

  signal S_AXIS_ACLK    : std_logic := '0';
  signal S_AXIS_ARESETN : std_logic := '0';
  signal S_AXIS_TREADY  : std_logic;
  signal S_AXIS_TDATA   : std_logic_vector(C_S_AXIS_TDATA_WIDTH - 1 downto 0)     := (others => '0');
  signal S_AXIS_TSTRB   : std_logic_vector((C_S_AXIS_TDATA_WIDTH/8) - 1 downto 0) := (others => '0');
  signal S_AXIS_TLAST   : std_logic                                               := '0';
  signal S_AXIS_TVALID  : std_logic                                               := '0';

  signal img_length_bytes : std_logic_vector(31 downto 0) := (others => '0');

  signal img_word_wr_en   : std_logic;
  signal img_word_wr_addr : unsigned(15 downto 0);
  signal img_word_wr_data : std_logic_vector(C_S_AXIS_TDATA_WIDTH - 1 downto 0);
  signal img_done         : std_logic;
  signal clear_img_done   : std_logic := '0';

begin

  -- DUT instance
  dut : entity work.MNIST_accel_slave_stream_v1_0_S00_AXIS
    generic map(
      C_S_AXIS_TDATA_WIDTH => C_S_AXIS_TDATA_WIDTH
    )
    port map
    (
      S_AXIS_ACLK    => S_AXIS_ACLK,
      S_AXIS_ARESETN => S_AXIS_ARESETN,
      S_AXIS_TREADY  => S_AXIS_TREADY,
      S_AXIS_TDATA   => S_AXIS_TDATA,
      S_AXIS_TSTRB   => S_AXIS_TSTRB,
      S_AXIS_TLAST   => S_AXIS_TLAST,
      S_AXIS_TVALID  => S_AXIS_TVALID,

      img_length_bytes => img_length_bytes,
      img_word_wr_en   => img_word_wr_en,
      img_word_wr_addr => img_word_wr_addr,
      img_word_wr_data => img_word_wr_data,
      img_done         => img_done,
      clear_img_done   => clear_img_done
    );

  -- Clock generation
  clk_gen : process
  begin
    while true loop
      S_AXIS_ACLK <= '0';
      wait for CLK_PERIOD/2;
      S_AXIS_ACLK <= '1';
      wait for CLK_PERIOD/2;
    end loop;
  end process clk_gen;

  -- Reset generation (active-low)
  rst_gen : process
  begin
    S_AXIS_ARESETN <= '0';
    wait for 5 * CLK_PERIOD;
    S_AXIS_ARESETN <= '1';
    wait;
  end process rst_gen;

  -- AXI-Stream stimulus
  stim_proc : process
    constant NUM_BEATS      : integer := 196; -- 784 bytes / 4 bytes per beat
    constant MAX_WAIT_READY : integer := 50;
    constant MAX_WAIT_DONE  : integer := 200;
    variable beat           : integer;
    variable wait_cnt       : integer;
  begin
    -- Wait reset deassertion
    wait until S_AXIS_ARESETN = '1';
    wait until rising_edge(S_AXIS_ACLK);

    -- Configure image length: 784 bytes
    img_length_bytes <= std_logic_vector(to_unsigned(784, img_length_bytes'length));

    -- Small delay to let DUT enter RX_IDLE
    wait for 5 * CLK_PERIOD;

    -- Initialize stream signals
    S_AXIS_TVALID <= '0';
    S_AXIS_TLAST  <= '0';
    S_AXIS_TSTRB  <= (others => '0');
    S_AXIS_TDATA  <= (others => '0');

    -- Send 196 beats
    for beat in 0 to NUM_BEATS - 1 loop
      S_AXIS_TDATA <= std_logic_vector(to_unsigned(beat, C_S_AXIS_TDATA_WIDTH));
      S_AXIS_TSTRB <= (S_AXIS_TSTRB'range => '1');

      if beat = NUM_BEATS - 1 then
        S_AXIS_TLAST <= '1';
      else
        S_AXIS_TLAST <= '0';
      end if;

      S_AXIS_TVALID <= '1';

      -- Wait for handshake with timeout
      wait_cnt := 0;
      while S_AXIS_TREADY = '0' loop
        wait until rising_edge(S_AXIS_ACLK);
        wait_cnt := wait_cnt + 1;
        if wait_cnt = MAX_WAIT_READY then
          assert false report "TREADY did not go high during frame transmission" severity error;
          exit;
        end if;
      end loop;

      -- Handshake at next clock edge
      wait until rising_edge(S_AXIS_ACLK);

      -- Deassert TVALID after beat
      S_AXIS_TVALID <= '0';
      S_AXIS_TLAST  <= '0';
      S_AXIS_TSTRB  <= (others => '0');

      -- Idle one cycle between beats
      wait until rising_edge(S_AXIS_ACLK);
    end loop;

    -- Ensure stream is idle
    S_AXIS_TVALID <= '0';
    S_AXIS_TLAST  <= '0';
    S_AXIS_TSTRB  <= (others => '0');
    S_AXIS_TDATA  <= (others => '0');

    -- Wait for img_done with timeout
    wait_cnt := 0;
    while img_done = '0' loop
      wait until rising_edge(S_AXIS_ACLK);
      wait_cnt := wait_cnt + 1;
      if wait_cnt = MAX_WAIT_DONE then
        assert false report "img_done did not assert after frame reception" severity error;
        exit;
      end if;
    end loop;

    -- Optional check on TREADY state after done
    assert S_AXIS_TREADY = '0'
    report "S_AXIS_TREADY should be '0' after img_done (RX_WAIT_CLEAR)"
      severity error;

    -- Clear img_done and check re-arm
    clear_img_done <= '1';
    wait until rising_edge(S_AXIS_ACLK);
    clear_img_done <= '0';

    wait_cnt := 0;
    while img_done = '1' loop
      wait until rising_edge(S_AXIS_ACLK);
      wait_cnt := wait_cnt + 1;
      if wait_cnt = MAX_WAIT_DONE then
        assert false report "img_done did not clear after clear_img_done pulse" severity error;
        exit;
      end if;
    end loop;

    wait_cnt := 0;
    while S_AXIS_TREADY = '0' loop
      wait until rising_edge(S_AXIS_ACLK);
      wait_cnt := wait_cnt + 1;
      if wait_cnt = MAX_WAIT_DONE then
        assert false report "S_AXIS_TREADY did not return to '1' after clear_img_done" severity error;
        exit;
      end if;
    end loop;

    report "AXI-Stream slave interface test completed successfully" severity note;
    wait for 10 * CLK_PERIOD;
    assert false report "End of simulation" severity failure;
  end process stim_proc;

end architecture sim;
