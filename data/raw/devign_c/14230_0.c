static uint64_t memcard_read(void *opaque, target_phys_addr_t addr,

                             unsigned size)

{

    MilkymistMemcardState *s = opaque;

    uint32_t r = 0;



    addr >>= 2;

    switch (addr) {

    case R_CMD:

        if (!s->enabled) {

            r = 0xff;

        } else {

            r = s->response[s->response_read_ptr++];

            if (s->response_read_ptr > s->response_len) {

                error_report("milkymist_memcard: "

                        "read more cmd bytes than available. Clipping.");

                s->response_read_ptr = 0;

            }

        }

        break;

    case R_DAT:

        if (!s->enabled) {

            r = 0xffffffff;

        } else {

            r = 0;

            r |= sd_read_data(s->card) << 24;

            r |= sd_read_data(s->card) << 16;

            r |= sd_read_data(s->card) << 8;

            r |= sd_read_data(s->card);

        }

        break;

    case R_CLK2XDIV:

    case R_ENABLE:

    case R_PENDING:

    case R_START:

        r = s->regs[addr];

        break;



    default:

        error_report("milkymist_memcard: read access to unknown register 0x"

                TARGET_FMT_plx, addr << 2);

        break;

    }



    trace_milkymist_memcard_memory_read(addr << 2, r);



    return r;

}
