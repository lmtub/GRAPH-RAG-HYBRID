MemoryRegion *iotlb_to_region(target_phys_addr_t index)

{

    return phys_sections[index & ~TARGET_PAGE_MASK].mr;

}
