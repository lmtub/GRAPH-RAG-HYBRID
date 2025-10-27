static int vhost_user_set_mem_table(struct vhost_dev *dev,

                                    struct vhost_memory *mem)

{

    int fds[VHOST_MEMORY_MAX_NREGIONS];

    int i, fd;

    size_t fd_num = 0;

    bool reply_supported = virtio_has_feature(dev->protocol_features,

                                              VHOST_USER_PROTOCOL_F_REPLY_ACK);



    VhostUserMsg msg = {

        .hdr.request = VHOST_USER_SET_MEM_TABLE,

        .hdr.flags = VHOST_USER_VERSION,

    };



    if (reply_supported) {

        msg.hdr.flags |= VHOST_USER_NEED_REPLY_MASK;

    }



    for (i = 0; i < dev->mem->nregions; ++i) {

        struct vhost_memory_region *reg = dev->mem->regions + i;

        ram_addr_t offset;

        MemoryRegion *mr;



        assert((uintptr_t)reg->userspace_addr == reg->userspace_addr);

        mr = memory_region_from_host((void *)(uintptr_t)reg->userspace_addr,

                                     &offset);

        fd = memory_region_get_fd(mr);

        if (fd > 0) {

            msg.payload.memory.regions[fd_num].userspace_addr = reg->userspace_addr;

            msg.payload.memory.regions[fd_num].memory_size  = reg->memory_size;

            msg.payload.memory.regions[fd_num].guest_phys_addr = reg->guest_phys_addr;

            msg.payload.memory.regions[fd_num].mmap_offset = offset;

            assert(fd_num < VHOST_MEMORY_MAX_NREGIONS);

            fds[fd_num++] = fd;

        }

    }



    msg.payload.memory.nregions = fd_num;



    if (!fd_num) {

        error_report("Failed initializing vhost-user memory map, "

                     "consider using -object memory-backend-file share=on");

        return -1;

    }



    msg.hdr.size = sizeof(msg.payload.memory.nregions);

    msg.hdr.size += sizeof(msg.payload.memory.padding);

    msg.hdr.size += fd_num * sizeof(VhostUserMemoryRegion);



    if (vhost_user_write(dev, &msg, fds, fd_num) < 0) {

        return -1;

    }



    if (reply_supported) {

        return process_message_reply(dev, &msg);

    }



    return 0;

}
