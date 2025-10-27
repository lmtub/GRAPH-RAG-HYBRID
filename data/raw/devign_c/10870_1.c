static int vhost_user_call(struct vhost_dev *dev, unsigned long int request,

        void *arg)

{

    VhostUserMsg msg;

    VhostUserRequest msg_request;

    struct vhost_vring_file *file = 0;

    int need_reply = 0;

    int fds[VHOST_MEMORY_MAX_NREGIONS];

    int i, fd;

    size_t fd_num = 0;



    assert(dev->vhost_ops->backend_type == VHOST_BACKEND_TYPE_USER);



    msg_request = vhost_user_request_translate(request);

    msg.request = msg_request;

    msg.flags = VHOST_USER_VERSION;

    msg.size = 0;



    switch (request) {

    case VHOST_GET_FEATURES:

        need_reply = 1;

        break;



    case VHOST_SET_FEATURES:

    case VHOST_SET_LOG_BASE:

        msg.u64 = *((__u64 *) arg);

        msg.size = sizeof(m.u64);

        break;



    case VHOST_SET_OWNER:

    case VHOST_RESET_OWNER:

        break;



    case VHOST_SET_MEM_TABLE:

        for (i = 0; i < dev->mem->nregions; ++i) {

            struct vhost_memory_region *reg = dev->mem->regions + i;

            ram_addr_t ram_addr;

            qemu_ram_addr_from_host((void *)reg->userspace_addr, &ram_addr);

            fd = qemu_get_ram_fd(ram_addr);

            if (fd > 0) {

                msg.memory.regions[fd_num].userspace_addr = reg->userspace_addr;

                msg.memory.regions[fd_num].memory_size  = reg->memory_size;

                msg.memory.regions[fd_num].guest_phys_addr = reg->guest_phys_addr;

                msg.memory.regions[fd_num].mmap_offset = reg->userspace_addr -

                    (uintptr_t) qemu_get_ram_block_host_ptr(reg->guest_phys_addr);

                assert(fd_num < VHOST_MEMORY_MAX_NREGIONS);

                fds[fd_num++] = fd;

            }

        }



        msg.memory.nregions = fd_num;



        if (!fd_num) {

            error_report("Failed initializing vhost-user memory map\n"

                    "consider using -object memory-backend-file share=on\n");

            return -1;

        }



        msg.size = sizeof(m.memory.nregions);

        msg.size += sizeof(m.memory.padding);

        msg.size += fd_num * sizeof(VhostUserMemoryRegion);



        break;



    case VHOST_SET_LOG_FD:

        fds[fd_num++] = *((int *) arg);

        break;



    case VHOST_SET_VRING_NUM:

    case VHOST_SET_VRING_BASE:

        memcpy(&msg.state, arg, sizeof(struct vhost_vring_state));

        msg.size = sizeof(m.state);

        break;



    case VHOST_GET_VRING_BASE:

        memcpy(&msg.state, arg, sizeof(struct vhost_vring_state));

        msg.size = sizeof(m.state);

        need_reply = 1;

        break;



    case VHOST_SET_VRING_ADDR:

        memcpy(&msg.addr, arg, sizeof(struct vhost_vring_addr));

        msg.size = sizeof(m.addr);

        break;



    case VHOST_SET_VRING_KICK:

    case VHOST_SET_VRING_CALL:

    case VHOST_SET_VRING_ERR:

        file = arg;

        msg.u64 = file->index & VHOST_USER_VRING_IDX_MASK;

        msg.size = sizeof(m.u64);

        if (ioeventfd_enabled() && file->fd > 0) {

            fds[fd_num++] = file->fd;

        } else {

            msg.u64 |= VHOST_USER_VRING_NOFD_MASK;

        }

        break;

    default:

        error_report("vhost-user trying to send unhandled ioctl\n");

        return -1;

        break;

    }



    if (vhost_user_write(dev, &msg, fds, fd_num) < 0) {

        return 0;

    }



    if (need_reply) {

        if (vhost_user_read(dev, &msg) < 0) {

            return 0;

        }



        if (msg_request != msg.request) {

            error_report("Received unexpected msg type."

                    " Expected %d received %d\n", msg_request, msg.request);

            return -1;

        }



        switch (msg_request) {

        case VHOST_USER_GET_FEATURES:

            if (msg.size != sizeof(m.u64)) {

                error_report("Received bad msg size.\n");

                return -1;

            }

            *((__u64 *) arg) = msg.u64;

            break;

        case VHOST_USER_GET_VRING_BASE:

            if (msg.size != sizeof(m.state)) {

                error_report("Received bad msg size.\n");

                return -1;

            }

            memcpy(arg, &msg.state, sizeof(struct vhost_vring_state));

            break;

        default:

            error_report("Received unexpected msg type.\n");

            return -1;

            break;

        }

    }



    return 0;

}
