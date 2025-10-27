static void xtensa_lx60_init(MachineState *machine)

{

    static const LxBoardDesc lx60_board = {

        .flash_base = 0xf8000000,

        .flash_size = 0x00400000,

        .flash_sector_size = 0x10000,

        .sram_size = 0x20000,

    };

    lx_init(&lx60_board, machine);

}
