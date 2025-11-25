library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity MNIST_accel_slave_stream_v1_0_S00_AXIS is
  generic (
    C_S_AXIS_TDATA_WIDTH : integer := 32
  );
  port (
    -- AXI4-Stream slave interface (from AXI DMA MM2S)
    S_AXIS_ACLK    : in std_logic;
    S_AXIS_ARESETN : in std_logic;
    S_AXIS_TREADY  : out std_logic;
    S_AXIS_TDATA   : in std_logic_vector(C_S_AXIS_TDATA_WIDTH - 1 downto 0);
    S_AXIS_TSTRB   : in std_logic_vector((C_S_AXIS_TDATA_WIDTH/8) - 1 downto 0);
    S_AXIS_TLAST   : in std_logic;
    S_AXIS_TVALID  : in std_logic;

    -- User-side interface: image buffer write + status
    -- Total image length in bytes (784 for MNIST)
    img_length_bytes : in std_logic_vector(31 downto 0);

    -- Write port for external image buffer
    -- One write per AXIS beat when TVALID & TREADY
    img_word_wr_en   : out std_logic;
    img_word_wr_addr : out unsigned(15 downto 0); -- up to 64K words
    img_word_wr_data : out std_logic_vector(C_S_AXIS_TDATA_WIDTH - 1 downto 0);

    -- Image done flag: asserted when full frame received
    img_done : out std_logic;
    -- Top/core asserts this to clear img_done and rearm the receiver
    clear_img_done : in std_logic
  );
end entity MNIST_accel_slave_stream_v1_0_S00_AXIS;

architecture arch_imp of MNIST_accel_slave_stream_v1_0_S00_AXIS is

  -- Internal constants
  constant BYTES_PER_BEAT : integer := C_S_AXIS_TDATA_WIDTH / 8;

  -- FSM for reception
  type rx_state_t is (RX_IDLE, RX_RECEIVE, RX_WAIT_CLEAR);
  signal rx_state : rx_state_t := RX_IDLE;

  -- Counters and flags
  signal word_count   : unsigned(15 downto 0) := (others => '0'); -- address for img_word_wr_addr
  signal byte_count   : unsigned(31 downto 0) := (others => '0'); -- total bytes received in this frame
  signal img_done_reg : std_logic             := '0';

  -- Registered TREADY
  signal s_axis_tready_reg : std_logic := '0';

begin

  -- AXIS outputs
  S_AXIS_TREADY <= s_axis_tready_reg;
  img_done      <= img_done_reg;

  -- Default assignments for buffer write signals (overridden in process)
  img_word_wr_en   <= '0';
  img_word_wr_addr <= word_count;
  img_word_wr_data <= S_AXIS_TDATA;

  -- Main receive process
  rx_proc : process (S_AXIS_ACLK)
    variable valid_bytes   : integer;
    variable target_length : unsigned(31 downto 0);
  begin
    if rising_edge(S_AXIS_ACLK) then
      if S_AXIS_ARESETN = '0' then
        rx_state          <= RX_IDLE;
        s_axis_tready_reg <= '0';
        word_count        <= (others => '0');
        byte_count        <= (others => '0');
        img_done_reg      <= '0';
        img_word_wr_en    <= '0'; -- add this line
      else
        img_word_wr_en <= '0'; -- default each cycle

        -- Cache desired length
        target_length := unsigned(img_length_bytes);

        case rx_state is
            -- RX_IDLE: waiting for first valid beat of a new frame.
          when RX_IDLE =>
            s_axis_tready_reg <= '1';
            img_done_reg      <= '0';
            word_count        <= (others => '0');
            byte_count        <= (others => '0');

            if (S_AXIS_TVALID = '1' and s_axis_tready_reg = '1') then
              -- Accept first beat
              img_word_wr_en <= '1';

              -- Compute number of valid bytes from TSTRB
              valid_bytes := 0;
              for i in 0 to BYTES_PER_BEAT - 1 loop
                if S_AXIS_TSTRB(i) = '1' then
                  valid_bytes := valid_bytes + 1;
                end if;
              end loop;

              byte_count <= unsigned(to_unsigned(valid_bytes, byte_count'length));
              word_count <= word_count + 1;

              -- Check if this beat already completes the frame
              if (S_AXIS_TLAST = '1') or
                (unsigned(to_unsigned(valid_bytes, byte_count'length)) >= target_length) then
                s_axis_tready_reg <= '0';
                img_done_reg      <= '1';
                rx_state          <= RX_WAIT_CLEAR;
              else
                rx_state <= RX_RECEIVE;
              end if;
            end if;

            -- RX_RECEIVE: streaming words into buffer until TLAST or
            --             img_length_bytes reached.
          when RX_RECEIVE =>
            s_axis_tready_reg <= '1';

            if (S_AXIS_TVALID = '1' and s_axis_tready_reg = '1') then
              -- Accept beat
              img_word_wr_en <= '1';

              -- Count valid bytes using TSTRB
              valid_bytes := 0;
              for i in 0 to BYTES_PER_BEAT - 1 loop
                if S_AXIS_TSTRB(i) = '1' then
                  valid_bytes := valid_bytes + 1;
                end if;
              end loop;

              byte_count <= byte_count + unsigned(to_unsigned(valid_bytes, byte_count'length));
              word_count <= word_count + 1;

              -- Check end-of-frame condition
              if (S_AXIS_TLAST = '1') or
                (byte_count + unsigned(to_unsigned(valid_bytes, byte_count'length)) >= target_length) then
                s_axis_tready_reg <= '0';
                img_done_reg      <= '1';
                rx_state          <= RX_WAIT_CLEAR;
              end if;
            end if;

            -- RX_WAIT_CLEAR: frame received; wait core/top to clear img_done
          when RX_WAIT_CLEAR =>
            s_axis_tready_reg <= '0'; -- ignore any incoming data until cleared

            if clear_img_done = '1' then
              img_done_reg <= '0';
              -- Ready to receive next image
              rx_state <= RX_IDLE;
            end if;

        end case;
      end if;
    end if;
  end process rx_proc;
end architecture arch_imp;
