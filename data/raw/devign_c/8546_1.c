static void digic_uart_write(void *opaque, hwaddr addr, uint64_t value,

                             unsigned size)

{

    DigicUartState *s = opaque;

    unsigned char ch = value;



    addr >>= 2;



    switch (addr) {

    case R_TX:

        if (s->chr) {



            qemu_chr_fe_write_all(s->chr, &ch, 1);

        }

        break;



    case R_ST:

        /*

         * Ignore write to R_ST.

         *

         * The point is that this register is actively used

         * during receiving and transmitting symbols,

         * but we don't know the function of most of bits.

         *

         * Ignoring writes to R_ST is only a simplification

         * of the model. It has no perceptible side effects

         * for existing guests.

         */

        break;



    default:

        qemu_log_mask(LOG_UNIMP,

                      "digic-uart: write access to unknown register 0x"

                      TARGET_FMT_plx, addr << 2);

    }

}