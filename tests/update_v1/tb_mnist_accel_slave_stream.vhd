library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity tb_mnist_accel_slave_stream is
end entity tb_mnist_accel_slave_stream;

architecture sim of tb_mnist_accel_slave_stream is

  constant C_S_AXIS_TDATA_WIDTH : integer := 32;
  constant CLK_PERIOD           : time    := 10 ns;

  -- DUT ports
  signal S_AXIS_ACLK    : std_logic := '0';
  signal S_AXIS_ARESETN : std_logic := '0';
  signal S_AXIS_TREADY  : std_logic;
  signal S_AXIS_TDATA   : std_logic_vector(C_S_AXIS_TDATA_WIDTH-1 downto 0) := (others => '0');
  signal S_AXIS_TSTRB   : std_logic_vector((C_S_AXIS_TDATA_WIDTH/8)-1 downto 0) := (others => '0');
  signal S_AXIS_TLAST   : std_logic := '0';
  signal S_AXIS_TVALID  : std_logic := '0';

  signal img_length_bytes : std_logic_vector(31 downto 0) := (others => '0');

  signal img_word_wr_en   : std_logic;
  signal img_word_wr_addr : unsigned(15 downto 0);
  signal img_word_wr_data : std_logic_vector(C_S_AXIS_TDATA_WIDTH-1 downto 0);
  signal img_done         : std_logic;
  signal clear_img_done   : std_logic := '0';

begin

  ---------------------------------------------------------------------------
  -- DUT instance
  ---------------------------------------------------------------------------
  dut : entity work.MNIST_accel_slave_stream_v1_0_S00_AXIS
    generic map (
      C_S_AXIS_TDATA_WIDTH => C_S_AXIS_TDATA_WIDTH
    )
    port map (
      S_AXIS_ACLK        => S_AXIS_ACLK,
      S_AXIS_ARESETN     => S_AXIS_ARESETN,
      S_AXIS_TREADY      => S_AXIS_TREADY,
      S_AXIS_TDATA       => S_AXIS_TDATA,
      S_AXIS_TSTRB       => S_AXIS_TSTRB,
      S_AXIS_TLAST       => S_AXIS_TLAST,
      S_AXIS_TVALID      => S_AXIS_TVALID,

      img_length_bytes   => img_length_bytes,
      img_word_wr_en     => img_word_wr_en,
      img_word_wr_addr   => img_word_wr_addr,
      img_word_wr_data   => img_word_wr_data,
      img_done           => img_done,
      clear_img_done     => clear_img_done
    );

  ---------------------------------------------------------------------------
  -- Clock generation
  ---------------------------------------------------------------------------
  clk_gen : process
  begin
    while true loop
      S_AXIS_ACLK <= '0';
      wait for CLK_PERIOD/2;
      S_AXIS_ACLK <= '1';
      wait for CLK_PERIOD/2;
    end loop;
  end process clk_gen;

  ---------------------------------------------------------------------------
  -- Reset generation (active-low)
  ---------------------------------------------------------------------------
  rst_gen : process
  begin
    S_AXIS_ARESETN <= '0';
    wait for 5*CLK_PERIOD;
    S_AXIS_ARESETN <= '1';
    wait;
  end process rst_gen;

  ---------------------------------------------------------------------------
  -- Stimulus: send one MNIST image (784 bytes = 196 words)
  ---------------------------------------------------------------------------
  stim_proc : process
    constant NUM_BEATS : integer := 196;
    variable beat_idx  : integer;
  begin
    -- Wait reset deassertion
    wait until S_AXIS_ARESETN = '1';
    wait until rising_edge(S_AXIS_ACLK);

    -- Configure image length: 784 bytes
    img_length_bytes <= std_logic_vector(to_unsigned(784, img_length_bytes'length));

    -- Pequeno tempo para DUT entrar em RX_IDLE
    wait for 5*CLK_PERIOD;

    -- Garante que TVALID/TLAST/TSTRB começam zerados
    S_AXIS_TVALID <= '0';
    S_AXIS_TLAST  <= '0';
    S_AXIS_TSTRB  <= (others => '0');
    S_AXIS_TDATA  <= (others => '0');

    -- Espera até o DUT ficar pronto (TREADY='1')
    wait until rising_edge(S_AXIS_ACLK) and S_AXIS_TREADY = '1';

    -------------------------------------------------------------------------
    -- Envia 196 beats com TSTRB=1111, TLAST no último
    -------------------------------------------------------------------------
    for beat_idx in 0 to NUM_BEATS-1 loop
      -- Prepara dados do beat
      S_AXIS_TVALID <= '1';
      S_AXIS_TSTRB  <= (S_AXIS_TSTRB'range => '1');  -- 0..3 => '1'
      S_AXIS_TDATA  <= std_logic_vector(to_unsigned(beat_idx, C_S_AXIS_TDATA_WIDTH));

      if beat_idx = NUM_BEATS-1 then
        S_AXIS_TLAST <= '1';
      else
        S_AXIS_TLAST <= '0';
      end if;

      -- Espera handshake (TVALID=1 e TREADY=1)
      loop
        wait until rising_edge(S_AXIS_ACLK);
        exit when (S_AXIS_TVALID = '1' and S_AXIS_TREADY = '1');
      end loop;

      -- Nesse ciclo, o beat foi aceito.
      -- Verifica se o endereço de escrita bate com o índice do beat.
      assert to_integer(img_word_wr_addr) = beat_idx
        report "img_word_wr_addr does not match expected beat index"
        severity error;

      -- (Opcional) se quiser conferir o dado:
      assert img_word_wr_data = S_AXIS_TDATA
        report "img_word_wr_data does not match S_AXIS_TDATA on accepted beat"
        severity error;
    end loop;

    -- Após enviar todos os beats, desativa TVALID/TLAST
    S_AXIS_TVALID <= '0';
    S_AXIS_TLAST  <= '0';
    S_AXIS_TSTRB  <= (others => '0');
    S_AXIS_TDATA  <= (others => '0');

    -- Espera alguns ciclos para o DUT fechar o frame
    wait for 5*CLK_PERIOD;

    -------------------------------------------------------------------------
    -- Checa que img_done subiu e TREADY foi desativado
    -------------------------------------------------------------------------
    assert img_done = '1'
      report "img_done was not asserted after full frame reception"
      severity error;

    assert S_AXIS_TREADY = '0'
      report "S_AXIS_TREADY should be '0' after frame done (RX_WAIT_CLEAR state)"
      severity error;

    -------------------------------------------------------------------------
    -- Exercita clear_img_done e verifica rearme
    -------------------------------------------------------------------------
    clear_img_done <= '1';
    wait until rising_edge(S_AXIS_ACLK);
    clear_img_done <= '0';

    -- Espera DUT voltar para RX_IDLE
    wait for 5*CLK_PERIOD;

    assert img_done = '0'
      report "img_done was not cleared after clear_img_done pulse"
      severity error;

    assert S_AXIS_TREADY = '1'
      report "S_AXIS_TREADY did not return to '1' after clear_img_done"
      severity error;

    -------------------------------------------------------------------------
    -- Fim da simulação
    -------------------------------------------------------------------------
    report "AXI-Stream slave interface test completed successfully" severity note;
    wait for 10*CLK_PERIOD;
    assert false report "End of simulation" severity failure;
  end process stim_proc;

end architecture sim;
