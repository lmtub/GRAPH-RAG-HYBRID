static int vtd_int_remap(X86IOMMUState *iommu, MSIMessage *src,

                         MSIMessage *dst, uint16_t sid)

{

    return vtd_interrupt_remap_msi(INTEL_IOMMU_DEVICE(iommu), src, dst);

}
