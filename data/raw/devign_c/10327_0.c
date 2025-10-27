static void ivshmem_io_writel(void *opaque, target_phys_addr_t addr,

                                                            uint32_t val)

{

    IVShmemState *s = opaque;



    uint64_t write_one = 1;

    uint16_t dest = val >> 16;

    uint16_t vector = val & 0xff;



    addr &= 0xfc;



    IVSHMEM_DPRINTF("writing to addr " TARGET_FMT_plx "\n", addr);

    switch (addr)

    {

        case INTRMASK:

            ivshmem_IntrMask_write(s, val);

            break;



        case INTRSTATUS:

            ivshmem_IntrStatus_write(s, val);

            break;



        case DOORBELL:

            /* check that dest VM ID is reasonable */

            if ((dest < 0) || (dest > s->max_peer)) {

                IVSHMEM_DPRINTF("Invalid destination VM ID (%d)\n", dest);

                break;

            }



            /* check doorbell range */

            if ((vector >= 0) && (vector < s->peers[dest].nb_eventfds)) {

                IVSHMEM_DPRINTF("Writing %" PRId64 " to VM %d on vector %d\n",

                                                    write_one, dest, vector);

                if (write(s->peers[dest].eventfds[vector],

                                                    &(write_one), 8) != 8) {

                    IVSHMEM_DPRINTF("error writing to eventfd\n");

                }

            }

            break;

        default:

            IVSHMEM_DPRINTF("Invalid VM Doorbell VM %d\n", dest);

    }

}
