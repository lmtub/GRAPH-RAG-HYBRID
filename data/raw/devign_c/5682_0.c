static uint64_t m5208_sys_read(void *opaque, target_phys_addr_t addr,

                               unsigned size)

{

    switch (addr) {

    case 0x110: /* SDCS0 */

        {

            int n;

            for (n = 0; n < 32; n++) {

                if (ram_size < (2u << n))

                    break;

            }

            return (n - 1)  | 0x40000000;

        }

    case 0x114: /* SDCS1 */

        return 0;



    default:

        hw_error("m5208_sys_read: Bad offset 0x%x\n", (int)addr);

        return 0;

    }

}
