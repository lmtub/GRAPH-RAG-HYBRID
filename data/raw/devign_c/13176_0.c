void sparc_iommu_memory_rw(void *opaque, target_phys_addr_t addr,

                           uint8_t *buf, int len, int is_write)

{

    int l;

    uint32_t flags;

    target_phys_addr_t page, phys_addr;



    while (len > 0) {

        page = addr & TARGET_PAGE_MASK;

        l = (page + TARGET_PAGE_SIZE) - addr;

        if (l > len)

            l = len;

        flags = iommu_page_get_flags(opaque, page);

        if (!(flags & IOPTE_VALID))

            return;

        phys_addr = iommu_translate_pa(opaque, addr, flags);

        if (is_write) {

            if (!(flags & IOPTE_WRITE))

                return;

            cpu_physical_memory_write(phys_addr, buf, len);

        } else {

            cpu_physical_memory_read(phys_addr, buf, len);

        }

        len -= l;

        buf += l;

        addr += l;

    }

}
