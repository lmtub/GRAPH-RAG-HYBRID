vubr_set_mem_table_exec(VubrDev *dev, VhostUserMsg *vmsg)

{

    int i;

    VhostUserMemory *memory = &vmsg->payload.memory;

    dev->nregions = memory->nregions;



    DPRINT("Nregions: %d\n", memory->nregions);

    for (i = 0; i < dev->nregions; i++) {

        void *mmap_addr;

        VhostUserMemoryRegion *msg_region = &memory->regions[i];

        VubrDevRegion *dev_region = &dev->regions[i];



        DPRINT("Region %d\n", i);

        DPRINT("    guest_phys_addr: 0x%016"PRIx64"\n",

               msg_region->guest_phys_addr);

        DPRINT("    memory_size:     0x%016"PRIx64"\n",

               msg_region->memory_size);

        DPRINT("    userspace_addr   0x%016"PRIx64"\n",

               msg_region->userspace_addr);

        DPRINT("    mmap_offset      0x%016"PRIx64"\n",

               msg_region->mmap_offset);



        dev_region->gpa = msg_region->guest_phys_addr;

        dev_region->size = msg_region->memory_size;

        dev_region->qva = msg_region->userspace_addr;

        dev_region->mmap_offset = msg_region->mmap_offset;



        /* We don't use offset argument of mmap() since the

         * mapped address has to be page aligned, and we use huge

         * pages.  */

        mmap_addr = mmap(0, dev_region->size + dev_region->mmap_offset,

                         PROT_READ | PROT_WRITE, MAP_SHARED,

                         vmsg->fds[i], 0);



        if (mmap_addr == MAP_FAILED) {

            vubr_die("mmap");

        }



        dev_region->mmap_addr = (uint64_t) mmap_addr;

        DPRINT("    mmap_addr:       0x%016"PRIx64"\n", dev_region->mmap_addr);

    }



    return 0;

}
