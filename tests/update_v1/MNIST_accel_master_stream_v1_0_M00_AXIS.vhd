library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity MNIST_accel_master_stream_v1_0_M00_AXIS is
  generic (
    C_M_AXIS_TDATA_WIDTH : integer := 32;
    C_M_AXIS_START_COUNT : integer := 32
  );
  port (
    -- AXI4-Stream master interface (to AXI DMA S2MM)
    M_AXIS_ACLK    : in  std_logic;
    M_AXIS_ARESETN : in  std_logic;
    M_AXIS_TVALID  : out std_logic;
    M_AXIS_TDATA   : out std_logic_vector(C_M_AXIS_TDATA_WIDTH-1 downto 0);
    M_AXIS_TSTRB   : out std_logic_vector((C_M_AXIS_TDATA_WIDTH/8)-1 downto 0);
    M_AXIS_TLAST   : out std_logic;
    M_AXIS_TREADY  : in  std_logic;

    -- User-side interface: logits vector + handshake
    -- 10 logits, each 8 bits, packed as:
    -- logits_data(7 downto 0)   = logit[0]
    -- logits_data(15 downto 8)  = logit[1]
    -- ...
    -- logits_data(79 downto 72) = logit[9]
    logits_data  : in  std_logic_vector(8*10-1 downto 0);
    -- Asserted by the core/top when logits_data is valid and stable
    logits_valid : in  std_logic;
    -- One-cycle pulse asserted when all logits have been sent
    logits_sent  : out std_logic
  );
end entity MNIST_accel_master_stream_v1_0_M00_AXIS;

architecture arch_imp of MNIST_accel_master_stream_v1_0_M00_AXIS is

  -- Internal constants
  constant LOGITS_PER_IMAGE : integer := 10;
  constant BYTES_PER_BEAT   : integer := C_M_AXIS_TDATA_WIDTH / 8;

  -- Number of AXIS beats needed to send all logits
  constant NUM_OUTPUT_BEATS : integer :=
    (LOGITS_PER_IMAGE + BYTES_PER_BEAT - 1) / BYTES_PER_BEAT;

  -- FSM for transmission
  type tx_state_t is (TX_IDLE, TX_SEND);
  signal tx_state : tx_state_t := TX_IDLE;

  -- Internal registers
  signal tvalid_reg : std_logic := '0';
  signal tlast_reg  : std_logic := '0';
  signal tdata_reg  : std_logic_vector(C_M_AXIS_TDATA_WIDTH-1 downto 0) := (others => '0');
  signal tstrb_reg  : std_logic_vector((C_M_AXIS_TDATA_WIDTH/8)-1 downto 0) := (others => '0');

  signal beat_index     : integer range 0 to NUM_OUTPUT_BEATS-1 := 0;
  signal logits_sent_reg: std_logic := '0';

begin

  -- AXIS outputs
  M_AXIS_TVALID <= tvalid_reg;
  M_AXIS_TDATA  <= tdata_reg;
  M_AXIS_TSTRB  <= tstrb_reg;
  M_AXIS_TLAST  <= tlast_reg;
  logits_sent   <= logits_sent_reg;

  -- Main transmit process
  tx_proc : process (M_AXIS_ACLK)
    -- Helper procedure to prepare TDATA and TSTRB for a given beat index
    procedure prepare_beat(
      constant beat   : in integer;
      signal   tdata  : out std_logic_vector;
      signal   tstrb  : out std_logic_vector
    ) is
      variable global_byte_index  : integer;
      variable bytes_in_this_beat : integer;
      variable i                  : integer;
      constant TOTAL_BYTES : integer := LOGITS_PER_IMAGE;
    begin
      -- Default zero
      tdata <= (others => '0');
      tstrb <= (others => '0');

      -- Determine how many bytes are valid in this beat
      if beat < NUM_OUTPUT_BEATS - 1 then
        -- All bytes valid for full beats
        bytes_in_this_beat := BYTES_PER_BEAT;
      else
        -- Last beat: remaining bytes
        bytes_in_this_beat := TOTAL_BYTES - (NUM_OUTPUT_BEATS - 1) * BYTES_PER_BEAT;
        if bytes_in_this_beat <= 0 then
          bytes_in_this_beat := BYTES_PER_BEAT; 
        end if;
      end if;

      -- Pack bytes from logits_data into TDATA
      for i in 0 to BYTES_PER_BEAT-1 loop
        global_byte_index := beat * BYTES_PER_BEAT + i;
        if global_byte_index < TOTAL_BYTES then
          -- Copy one logit byte
          tdata(8*i+7 downto 8*i) <=
logits_data(8*global_byte_index+7 downto 8*global_byte_index);
        else
          tdata(8*i+7 downto 8*i) <= (others => '0');
        end if;

        -- Set TSTRB bit if this byte is valid in this beat
        if i < bytes_in_this_beat then
          tstrb(i) <= '1';
        else
          tstrb(i) <= '0';
        end if;
      end loop;
    end procedure;
  begin
    if rising_edge(M_AXIS_ACLK) then
      if M_AXIS_ARESETN = '0' then
        tx_state        <= TX_IDLE;
        tvalid_reg      <= '0';
        tlast_reg       <= '0';
        tdata_reg       <= (others => '0');
        tstrb_reg       <= (others => '0');
        beat_index      <= 0;
        logits_sent_reg <= '0';
      else
        -- Default: logits_sent is a one-cycle pulse
        logits_sent_reg <= '0';

        case tx_state is

          -- TX_IDLE: wait for logits_valid from core/top.
          when TX_IDLE =>
            tvalid_reg <= '0';
            tlast_reg  <= '0';
            beat_index <= 0;

            if logits_valid = '1' then
              -- Prepare first beat
              prepare_beat(0, tdata_reg, tstrb_reg);
              tvalid_reg <= '1';

              -- If only one beat is needed, TLAST is asserted immediately
              if NUM_OUTPUT_BEATS = 1 then
                tlast_reg <= '1';
              else
                tlast_reg <= '0';
              end if;

              tx_state <= TX_SEND;
            end if;

          -- TX_SEND: send beats while TVALID & TREADY handshaking.
          when TX_SEND =>
            if tvalid_reg = '1' and M_AXIS_TREADY = '1' then
              -- Current beat has been accepted
              if beat_index = NUM_OUTPUT_BEATS - 1 then
                -- Last beat sent
                tvalid_reg      <= '0';
                tlast_reg       <= '0';
                logits_sent_reg <= '1';  -- notify top/core that transmission is complete
                tx_state        <= TX_IDLE;
              else
                -- Prepare next beat
                beat_index <= beat_index + 1;
                prepare_beat(beat_index + 1, tdata_reg, tstrb_reg);

                if (beat_index + 1) = NUM_OUTPUT_BEATS - 1 then
                  tlast_reg <= '1';
                else
                  tlast_reg <= '0';
                end if;
              end if;
            end if;

        end case;
      end if;
    end if;
  end process tx_proc;
end architecture arch_imp;
