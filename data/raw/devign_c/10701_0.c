int vfio_region_mmap(VFIORegion *region)

{

    int i, prot = 0;

    char *name;



    if (!region->mem) {

        return 0;

    }



    prot |= region->flags & VFIO_REGION_INFO_FLAG_READ ? PROT_READ : 0;

    prot |= region->flags & VFIO_REGION_INFO_FLAG_WRITE ? PROT_WRITE : 0;



    for (i = 0; i < region->nr_mmaps; i++) {

        region->mmaps[i].mmap = mmap(NULL, region->mmaps[i].size, prot,

                                     MAP_SHARED, region->vbasedev->fd,

                                     region->fd_offset +

                                     region->mmaps[i].offset);

        if (region->mmaps[i].mmap == MAP_FAILED) {

            int ret = -errno;



            trace_vfio_region_mmap_fault(memory_region_name(region->mem), i,

                                         region->fd_offset +

                                         region->mmaps[i].offset,

                                         region->fd_offset +

                                         region->mmaps[i].offset +

                                         region->mmaps[i].size - 1, ret);



            region->mmaps[i].mmap = NULL;



            for (i--; i >= 0; i--) {

                memory_region_del_subregion(region->mem, &region->mmaps[i].mem);

                munmap(region->mmaps[i].mmap, region->mmaps[i].size);

                object_unparent(OBJECT(&region->mmaps[i].mem));

                region->mmaps[i].mmap = NULL;

            }



            return ret;

        }



        name = g_strdup_printf("%s mmaps[%d]",

                               memory_region_name(region->mem), i);

        memory_region_init_ram_ptr(&region->mmaps[i].mem,

                                   memory_region_owner(region->mem),

                                   name, region->mmaps[i].size,

                                   region->mmaps[i].mmap);

        g_free(name);

        memory_region_set_skip_dump(&region->mmaps[i].mem);

        memory_region_add_subregion(region->mem, region->mmaps[i].offset,

                                    &region->mmaps[i].mem);



        trace_vfio_region_mmap(memory_region_name(&region->mmaps[i].mem),

                               region->mmaps[i].offset,

                               region->mmaps[i].offset +

                               region->mmaps[i].size - 1);

    }



    return 0;

}
