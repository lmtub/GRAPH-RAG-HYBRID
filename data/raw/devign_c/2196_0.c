static void vfio_unmap_bar(VFIOPCIDevice *vdev, int nr)

{

    VFIOBAR *bar = &vdev->bars[nr];



    if (!bar->region.size) {

        return;

    }



    vfio_bar_quirk_teardown(vdev, nr);



    memory_region_del_subregion(&bar->region.mem, &bar->region.mmap_mem);

    munmap(bar->region.mmap, memory_region_size(&bar->region.mmap_mem));



    if (vdev->msix && vdev->msix->table_bar == nr) {

        memory_region_del_subregion(&bar->region.mem, &vdev->msix->mmap_mem);

        munmap(vdev->msix->mmap, memory_region_size(&vdev->msix->mmap_mem));

    }

}
