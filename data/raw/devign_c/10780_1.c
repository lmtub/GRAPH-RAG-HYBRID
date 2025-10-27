static void spapr_phb_vfio_finish_realize(sPAPRPHBState *sphb, Error **errp)

{

    sPAPRPHBVFIOState *svphb = SPAPR_PCI_VFIO_HOST_BRIDGE(sphb);

    struct vfio_iommu_spapr_tce_info info = { .argsz = sizeof(info) };

    int ret;

    sPAPRTCETable *tcet;

    uint32_t liobn = svphb->phb.dma_liobn;



    if (svphb->iommugroupid == -1) {

        error_setg(errp, "Wrong IOMMU group ID %d", svphb->iommugroupid);

        return;

    }



    ret = vfio_container_ioctl(&svphb->phb.iommu_as, svphb->iommugroupid,

                               VFIO_CHECK_EXTENSION,

                               (void *) VFIO_SPAPR_TCE_IOMMU);

    if (ret != 1) {

        error_setg_errno(errp, -ret,

                         "spapr-vfio: SPAPR extension is not supported");

        return;

    }



    ret = vfio_container_ioctl(&svphb->phb.iommu_as, svphb->iommugroupid,

                               VFIO_IOMMU_SPAPR_TCE_GET_INFO, &info);

    if (ret) {

        error_setg_errno(errp, -ret,

                         "spapr-vfio: get info from container failed");

        return;

    }



    tcet = spapr_tce_new_table(DEVICE(sphb), liobn, info.dma32_window_start,

                               SPAPR_TCE_PAGE_SHIFT,

                               info.dma32_window_size >> SPAPR_TCE_PAGE_SHIFT,

                               true);

    if (!tcet) {

        error_setg(errp, "spapr-vfio: failed to create VFIO TCE table");

        return;

    }



    /* Register default 32bit DMA window */

    memory_region_add_subregion(&sphb->iommu_root, tcet->bus_offset,

                                spapr_tce_get_iommu(tcet));

}
