static uint64_t ivshmem_io_read(void *opaque, hwaddr addr,

                                unsigned size)

{



    IVShmemState *s = opaque;

    uint32_t ret;



    switch (addr)

    {

        case INTRMASK:

            ret = ivshmem_IntrMask_read(s);

            break;



        case INTRSTATUS:

            ret = ivshmem_IntrStatus_read(s);

            break;



        case IVPOSITION:

            /* return my VM ID if the memory is mapped */

            if (s->shm_fd >= 0) {

                ret = s->vm_id;

            } else {

                ret = -1;

            }

            break;



        default:

            IVSHMEM_DPRINTF("why are we reading " TARGET_FMT_plx "\n", addr);

            ret = 0;

    }



    return ret;

}
