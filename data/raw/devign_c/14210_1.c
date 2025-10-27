static void ivshmem_read(void *opaque, const uint8_t *buf, int size)

{

    IVShmemState *s = opaque;

    int incoming_fd, tmp_fd;

    int guest_max_eventfd;

    long incoming_posn;



    if (fifo8_is_empty(&s->incoming_fifo) && size == sizeof(incoming_posn)) {

        memcpy(&incoming_posn, buf, size);

    } else {

        const uint8_t *p;

        uint32_t num;



        IVSHMEM_DPRINTF("short read of %d bytes\n", size);

        num = MAX(size, sizeof(long) - fifo8_num_used(&s->incoming_fifo));

        fifo8_push_all(&s->incoming_fifo, buf, num);

        if (fifo8_num_used(&s->incoming_fifo) < sizeof(incoming_posn)) {

            return;

        }

        size -= num;

        buf += num;

        p = fifo8_pop_buf(&s->incoming_fifo, sizeof(incoming_posn), &num);

        g_assert(num == sizeof(incoming_posn));

        memcpy(&incoming_posn, p, sizeof(incoming_posn));

        if (size > 0) {

            fifo8_push_all(&s->incoming_fifo, buf, size);

        }

    }



    if (incoming_posn < -1) {

        IVSHMEM_DPRINTF("invalid incoming_posn %ld\n", incoming_posn);

        return;

    }



    /* pick off s->server_chr->msgfd and store it, posn should accompany msg */

    tmp_fd = qemu_chr_fe_get_msgfd(s->server_chr);

    IVSHMEM_DPRINTF("posn is %ld, fd is %d\n", incoming_posn, tmp_fd);



    /* make sure we have enough space for this guest */

    if (incoming_posn >= s->nb_peers) {

        if (increase_dynamic_storage(s, incoming_posn) < 0) {

            error_report("increase_dynamic_storage() failed");

            if (tmp_fd != -1) {


            }

            return;

        }

    }



    if (tmp_fd == -1) {

        /* if posn is positive and unseen before then this is our posn*/

        if ((incoming_posn >= 0) &&

                            (s->peers[incoming_posn].eventfds == NULL)) {

            /* receive our posn */

            s->vm_id = incoming_posn;

            return;

        } else {

            /* otherwise an fd == -1 means an existing guest has gone away */

            IVSHMEM_DPRINTF("posn %ld has gone away\n", incoming_posn);

            close_guest_eventfds(s, incoming_posn);

            return;

        }

    }



    /* because of the implementation of get_msgfd, we need a dup */

    incoming_fd = dup(tmp_fd);



    if (incoming_fd == -1) {

        fprintf(stderr, "could not allocate file descriptor %s\n",

                                                            strerror(errno));


        return;

    }



    /* if the position is -1, then it's shared memory region fd */

    if (incoming_posn == -1) {



        void * map_ptr;



        s->max_peer = 0;



        if (check_shm_size(s, incoming_fd) == -1) {

            exit(-1);

        }



        /* mmap the region and map into the BAR2 */

        map_ptr = mmap(0, s->ivshmem_size, PROT_READ|PROT_WRITE, MAP_SHARED,

                                                            incoming_fd, 0);

        memory_region_init_ram_ptr(&s->ivshmem, OBJECT(s),

                                   "ivshmem.bar2", s->ivshmem_size, map_ptr);

        vmstate_register_ram(&s->ivshmem, DEVICE(s));



        IVSHMEM_DPRINTF("guest h/w addr = %p, size = %" PRIu64 "\n",

                         map_ptr, s->ivshmem_size);



        memory_region_add_subregion(&s->bar, 0, &s->ivshmem);



        /* only store the fd if it is successfully mapped */

        s->shm_fd = incoming_fd;



        return;

    }



    /* each guest has an array of eventfds, and we keep track of how many

     * guests for each VM */

    guest_max_eventfd = s->peers[incoming_posn].nb_eventfds;



    if (guest_max_eventfd == 0) {

        /* one eventfd per MSI vector */

        s->peers[incoming_posn].eventfds = g_new(EventNotifier, s->vectors);

    }



    /* this is an eventfd for a particular guest VM */

    IVSHMEM_DPRINTF("eventfds[%ld][%d] = %d\n", incoming_posn,

                                            guest_max_eventfd, incoming_fd);

    event_notifier_init_fd(&s->peers[incoming_posn].eventfds[guest_max_eventfd],

                           incoming_fd);



    /* increment count for particular guest */

    s->peers[incoming_posn].nb_eventfds++;



    /* keep track of the maximum VM ID */

    if (incoming_posn > s->max_peer) {

        s->max_peer = incoming_posn;

    }



    if (incoming_posn == s->vm_id) {

        s->eventfd_chr[guest_max_eventfd] = create_eventfd_chr_device(s,

                   &s->peers[s->vm_id].eventfds[guest_max_eventfd],

                   guest_max_eventfd);

    }



    if (ivshmem_has_feature(s, IVSHMEM_IOEVENTFD)) {

        ivshmem_add_eventfd(s, incoming_posn, guest_max_eventfd);

    }

}