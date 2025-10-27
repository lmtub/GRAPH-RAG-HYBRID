static void qemu_remap_bucket(MapCacheEntry *entry,

                              target_phys_addr_t size,

                              target_phys_addr_t address_index)

{

    uint8_t *vaddr_base;

    xen_pfn_t *pfns;

    int *err;

    unsigned int i, j;

    target_phys_addr_t nb_pfn = size >> XC_PAGE_SHIFT;



    trace_qemu_remap_bucket(address_index);



    pfns = qemu_mallocz(nb_pfn * sizeof (xen_pfn_t));

    err = qemu_mallocz(nb_pfn * sizeof (int));



    if (entry->vaddr_base != NULL) {

        if (munmap(entry->vaddr_base, size) != 0) {

            perror("unmap fails");

            exit(-1);

        }

    }



    for (i = 0; i < nb_pfn; i++) {

        pfns[i] = (address_index << (MCACHE_BUCKET_SHIFT-XC_PAGE_SHIFT)) + i;

    }



    vaddr_base = xc_map_foreign_bulk(xen_xc, xen_domid, PROT_READ|PROT_WRITE,

                                     pfns, err, nb_pfn);

    if (vaddr_base == NULL) {

        perror("xc_map_foreign_bulk");

        exit(-1);

    }



    entry->vaddr_base = vaddr_base;

    entry->paddr_index = address_index;



    for (i = 0; i < nb_pfn; i += BITS_PER_LONG) {

        unsigned long word = 0;

        if ((i + BITS_PER_LONG) > nb_pfn) {

            j = nb_pfn % BITS_PER_LONG;

        } else {

            j = BITS_PER_LONG;

        }

        while (j > 0) {

            word = (word << 1) | !err[i + --j];

        }

        entry->valid_mapping[i / BITS_PER_LONG] = word;

    }



    qemu_free(pfns);

    qemu_free(err);

}
