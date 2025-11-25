library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity MNIST_accel_slave_lite_v1_0_S00_AXI is
  generic (
    -- Width of S_AXI data bus
    C_S_AXI_DATA_WIDTH : integer := 32;
    -- Width of S_AXI address bus
    C_S_AXI_ADDR_WIDTH : integer := 4
  );
  port (
    -- User ports (interface to core / top-level)
    -- One-cycle start pulse generated on write to CTRL bit[0] = '1'
    ctrl_start_pulse : out std_logic;
    -- Latched interrupt enable from CTRL bit[1]
    ctrl_irq_en      : out std_logic;
    -- Status bits driven by the core
    status_busy      : in  std_logic;
    status_done      : in  std_logic;
    status_error     : in  std_logic := '0';
    -- Image length in bytes (configurable via AXI, default 784)
    img_length       : out std_logic_vector(C_S_AXI_DATA_WIDTH-1 downto 0);

    -- AXI4-Lite slave interface
    -- Global Clock Signal
    S_AXI_ACLK    : in  std_logic;
    -- Global Reset Signal. This Signal is Active LOW
    S_AXI_ARESETN : in  std_logic;
    -- Write address (issued by master, accepted by Slave)
    S_AXI_AWADDR  : in  std_logic_vector(C_S_AXI_ADDR_WIDTH-1 downto 0);
    -- Write address protection type
    S_AXI_AWPROT  : in  std_logic_vector(2 downto 0);
    -- Write address valid
    S_AXI_AWVALID : in  std_logic;
    -- Write address ready
    S_AXI_AWREADY : out std_logic;
    -- Write data
    S_AXI_WDATA   : in  std_logic_vector(C_S_AXI_DATA_WIDTH-1 downto 0);
    -- Write strobes
    S_AXI_WSTRB   : in  std_logic_vector((C_S_AXI_DATA_WIDTH/8)-1 downto 0);
    -- Write valid
    S_AXI_WVALID  : in  std_logic;
    -- Write ready
    S_AXI_WREADY  : out std_logic;
    -- Write response
    S_AXI_BRESP   : out std_logic_vector(1 downto 0);
    -- Write response valid
    S_AXI_BVALID  : out std_logic;
    -- Response ready
    S_AXI_BREADY  : in  std_logic;
    -- Read address (issued by master, accepted by Slave)
    S_AXI_ARADDR  : in  std_logic_vector(C_S_AXI_ADDR_WIDTH-1 downto 0);
    -- Protection type
    S_AXI_ARPROT  : in  std_logic_vector(2 downto 0);
    -- Read address valid
    S_AXI_ARVALID : in  std_logic;
    -- Read address ready
    S_AXI_ARREADY : out std_logic;
    -- Read data
    S_AXI_RDATA   : out std_logic_vector(C_S_AXI_DATA_WIDTH-1 downto 0);
    -- Read response
    S_AXI_RRESP   : out std_logic_vector(1 downto 0);
    -- Read valid
    S_AXI_RVALID  : out std_logic;
    -- Read ready
    S_AXI_RREADY  : in  std_logic
  );
end entity MNIST_accel_slave_lite_v1_0_S00_AXI;

architecture arch_imp of MNIST_accel_slave_lite_v1_0_S00_AXI is

  -- AXI4-Lite signals
  signal axi_awaddr  : std_logic_vector(C_S_AXI_ADDR_WIDTH-1 downto 0);
  signal axi_awready : std_logic;
  signal axi_wready  : std_logic;
  signal axi_bresp   : std_logic_vector(1 downto 0);
  signal axi_bvalid  : std_logic;
  signal axi_araddr  : std_logic_vector(C_S_AXI_ADDR_WIDTH-1 downto 0);
  signal axi_arready : std_logic;
  signal axi_rdata   : std_logic_vector(C_S_AXI_DATA_WIDTH-1 downto 0);
  signal axi_rresp   : std_logic_vector(1 downto 0);
  signal axi_rvalid  : std_logic;

  -- Addressing parameters
  -- For 32-bit data: ADDR_LSB = 2 -> word aligned
  constant ADDR_LSB           : integer := (C_S_AXI_DATA_WIDTH/32) + 1;
  constant OPT_MEM_ADDR_BITS  : integer := C_S_AXI_ADDR_WIDTH - ADDR_LSB;

  -- Register address decoding (integer indices for case statements)
  signal awaddr_index : integer range 0 to 2**OPT_MEM_ADDR_BITS - 1;
  signal araddr_index : integer range 0 to 2**OPT_MEM_ADDR_BITS - 1;

  -- Internal registers
  -- CTRL register (offset 0x00)
  -- bit 0: START (edge-triggered, generates ctrl_start_pulse)
  -- bit 1: IRQ_EN (latched)
  signal reg_ctrl      : std_logic_vector(C_S_AXI_DATA_WIDTH-1 downto 0);

  -- IMG_LENGTH register (offset 0x08)
  -- Number of image bytes to be ingested by AXI-Stream slave
  signal reg_img_length : std_logic_vector(C_S_AXI_DATA_WIDTH-1 downto 0);

  -- Latched interrupt enable
  signal irq_en_reg      : std_logic;
  -- One-cycle start pulse register
  signal start_pulse_reg : std_logic;

  -- Read data mux
  signal reg_data_out : std_logic_vector(C_S_AXI_DATA_WIDTH-1 downto 0);

  -- Write enable for registers
  signal slv_reg_wren : std_logic;
  -- Read enable
  signal slv_reg_rden : std_logic;

begin

  -----------------------------------------------------------------------------
  -- AXI4-Lite interface assignments
  -----------------------------------------------------------------------------
  S_AXI_AWREADY <= axi_awready;
  S_AXI_WREADY  <= axi_wready;
  S_AXI_BRESP   <= axi_bresp;
  S_AXI_BVALID  <= axi_bvalid;
  S_AXI_ARREADY <= axi_arready;
  S_AXI_RDATA   <= axi_rdata;
  S_AXI_RRESP   <= axi_rresp;
  S_AXI_RVALID  <= axi_rvalid;

  -- User outputs
  ctrl_start_pulse <= start_pulse_reg;
  ctrl_irq_en      <= irq_en_reg;
  img_length       <= reg_img_length;

  -- Decode address indices (word address)
  awaddr_index <= to_integer(
                    unsigned(
                      axi_awaddr(ADDR_LSB+OPT_MEM_ADDR_BITS-1 downto ADDR_LSB)
                    )
                  );
  araddr_index <= to_integer(
                    unsigned(
                      axi_araddr(ADDR_LSB+OPT_MEM_ADDR_BITS-1 downto ADDR_LSB)
                    )
                  );

  -----------------------------------------------------------------------------
  -- Write address channel
  -----------------------------------------------------------------------------
  process (S_AXI_ACLK)
  begin
    if rising_edge(S_AXI_ACLK) then
      if S_AXI_ARESETN = '0' then
        axi_awready <= '0';
        axi_awaddr  <= (others => '0');
      else
        if (axi_awready = '0' and S_AXI_AWVALID = '1' and S_AXI_WVALID = '1') then
          axi_awready <= '1';
          axi_awaddr  <= S_AXI_AWADDR;
        else
          axi_awready <= '0';
        end if;
      end if;
    end if;
  end process;

  -----------------------------------------------------------------------------
  -- Write data channel
  -----------------------------------------------------------------------------
  process (S_AXI_ACLK)
  begin
    if rising_edge(S_AXI_ACLK) then
      if S_AXI_ARESETN = '0' then
        axi_wready <= '0';
      else
        if (axi_wready = '0' and S_AXI_WVALID = '1' and S_AXI_AWVALID = '1') then
          axi_wready <= '1';
        else
          axi_wready <= '0';
        end if;
      end if;
    end if;
  end process;

  -----------------------------------------------------------------------------
  -- Write response channel
  -----------------------------------------------------------------------------
  process (S_AXI_ACLK)
  begin
    if rising_edge(S_AXI_ACLK) then
      if S_AXI_ARESETN = '0' then
        axi_bvalid <= '0';
        axi_bresp  <= (others => '0');
      else
        if (axi_awready = '1' and S_AXI_AWVALID = '1' and
            axi_wready  = '1' and S_AXI_WVALID  = '1' and
            axi_bvalid  = '0') then
          axi_bvalid <= '1';
          axi_bresp  <= "00";  -- OKAY response
        elsif (S_AXI_BREADY = '1' and axi_bvalid = '1') then
          axi_bvalid <= '0';
        end if;
      end if;
    end if;
  end process;

  -- Register write enable: AW + W handshake
  slv_reg_wren <= '1' when (axi_awready = '1' and S_AXI_AWVALID = '1' and
                            axi_wready  = '1' and S_AXI_WVALID  = '1')
                  else '0';

  -----------------------------------------------------------------------------
  -- Read address channel
  -----------------------------------------------------------------------------
  process (S_AXI_ACLK)
  begin
    if rising_edge(S_AXI_ACLK) then
      if S_AXI_ARESETN = '0' then
        axi_arready <= '0';
        axi_araddr  <= (others => '0');
      else
        if (axi_arready = '0' and S_AXI_ARVALID = '1') then
          axi_arready <= '1';
          axi_araddr  <= S_AXI_ARADDR;
        else
          axi_arready <= '0';
        end if;
      end if;
    end if;
  end process;

  -- Read enable: AR handshake
  slv_reg_rden <= '1' when (axi_arready = '1' and S_AXI_ARVALID = '1' and
                            axi_rvalid  = '0')
                  else '0';

  -----------------------------------------------------------------------------
  -- Read data channel
  -----------------------------------------------------------------------------
  process (S_AXI_ACLK)
  begin
    if rising_edge(S_AXI_ACLK) then
      if S_AXI_ARESETN = '0' then
        axi_rvalid <= '0';
        axi_rresp  <= (others => '0');
      else
        if (slv_reg_rden = '1') then
          axi_rvalid <= '1';
          axi_rresp  <= "00";  -- OKAY
        elsif (axi_rvalid = '1' and S_AXI_RREADY = '1') then
          axi_rvalid <= '0';
        end if;
      end if;
    end if;
  end process;

  -----------------------------------------------------------------------------
  -- Register write logic (CTRL, IMG_LENGTH, IRQ_EN, START pulse)
  -----------------------------------------------------------------------------
  process (S_AXI_ACLK)
  begin
    if rising_edge(S_AXI_ACLK) then
      if S_AXI_ARESETN = '0' then
        reg_ctrl       <= (others => '0');
        reg_img_length <= (others => '0');
        irq_en_reg     <= '0';
      else
        -- default: keep previous register values
        if slv_reg_wren = '1' then
          case awaddr_index is

            -- CTRL register at 0x00
            when 0 =>
              for byte_index in 0 to (C_S_AXI_DATA_WIDTH/8 - 1) loop
                if S_AXI_WSTRB(byte_index) = '1' then
                  reg_ctrl(byte_index*8+7 downto byte_index*8) <=
                    S_AXI_WDATA(byte_index*8+7 downto byte_index*8);
                end if;
              end loop;
              -- Update IRQ_EN bit from reg_ctrl(1)
              irq_en_reg <= S_AXI_WDATA(1);

            -- STATUS register at 0x04 is READ-ONLY
            -- Writes are ignored intentionally
            when 1 =>
              null;

            -- IMG_LENGTH register at 0x08
            when 2 =>
              for byte_index in 0 to (C_S_AXI_DATA_WIDTH/8 - 1) loop
                if S_AXI_WSTRB(byte_index) = '1' then
                  reg_img_length(byte_index*8+7 downto byte_index*8) <=
                    S_AXI_WDATA(byte_index*8+7 downto byte_index*8);
                end if;
              end loop;

            when others =>
              null;

          end case;
        end if;
      end if;
    end if;
  end process;

  -----------------------------------------------------------------------------
  -- START pulse generation (one cycle)
  -----------------------------------------------------------------------------
  process (S_AXI_ACLK)
  begin
    if rising_edge(S_AXI_ACLK) then
      if S_AXI_ARESETN = '0' then
        start_pulse_reg <= '0';
      else
        -- default: no pulse
        start_pulse_reg <= '0';

        if slv_reg_wren = '1' and awaddr_index = 0 then
          -- Only consider LSB (bit 0) and its byte strobe
          if S_AXI_WSTRB(0) = '1' and S_AXI_WDATA(0) = '1' then
            start_pulse_reg <= '1';
          end if;
        end if;
      end if;
    end if;
  end process;

  -----------------------------------------------------------------------------
  -- Read data mux (CTRL, STATUS, IMG_LENGTH)
  -----------------------------------------------------------------------------
  -- STATUS is read-only and driven from top-level:
  -- bit 0: DONE
  -- bit 1: BUSY
  -- bit 2: ERROR
  process (araddr_index, reg_ctrl, reg_img_length,
           status_busy, status_done, status_error)
  begin
    reg_data_out <= (others => '0');

    case araddr_index is
      -- CTRL (readback)
      when 0 =>
        reg_data_out <= reg_ctrl;

      -- STATUS (DONE / BUSY / ERROR)
      when 1 =>
        reg_data_out(0) <= status_done;
        reg_data_out(1) <= status_busy;
        reg_data_out(2) <= status_error;
        -- remaining bits left as '0'

      -- IMG_LENGTH
      when 2 =>
        reg_data_out <= reg_img_length;

      when others =>
        reg_data_out <= (others => '0');
    end case;
  end process;

  -- Drive AXI read data
  axi_rdata <= reg_data_out;

end architecture arch_imp;
