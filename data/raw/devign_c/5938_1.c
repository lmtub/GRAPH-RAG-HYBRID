static int ioreq_map(struct ioreq *ioreq)

{

    XenGnttab gnt = ioreq->blkdev->xendev.gnttabdev;

    uint32_t domids[BLKIF_MAX_SEGMENTS_PER_REQUEST];

    uint32_t refs[BLKIF_MAX_SEGMENTS_PER_REQUEST];

    void *page[BLKIF_MAX_SEGMENTS_PER_REQUEST];

    int i, j, new_maps = 0;

    PersistentGrant *grant;

    /* domids and refs variables will contain the information necessary

     * to map the grants that are needed to fulfill this request.

     *

     * After mapping the needed grants, the page array will contain the

     * memory address of each granted page in the order specified in ioreq

     * (disregarding if it's a persistent grant or not).

     */



    if (ioreq->v.niov == 0 || ioreq->mapped == 1) {

        return 0;

    }

    if (ioreq->blkdev->feature_persistent) {

        for (i = 0; i < ioreq->v.niov; i++) {

            grant = g_tree_lookup(ioreq->blkdev->persistent_gnts,

                                    GUINT_TO_POINTER(ioreq->refs[i]));



            if (grant != NULL) {

                page[i] = grant->page;

                xen_be_printf(&ioreq->blkdev->xendev, 3,

                              "using persistent-grant %" PRIu32 "\n",

                              ioreq->refs[i]);

            } else {

                    /* Add the grant to the list of grants that

                     * should be mapped

                     */

                    domids[new_maps] = ioreq->domids[i];

                    refs[new_maps] = ioreq->refs[i];

                    page[i] = NULL;

                    new_maps++;

            }

        }

        /* Set the protection to RW, since grants may be reused later

         * with a different protection than the one needed for this request

         */

        ioreq->prot = PROT_WRITE | PROT_READ;

    } else {

        /* All grants in the request should be mapped */

        memcpy(refs, ioreq->refs, sizeof(refs));

        memcpy(domids, ioreq->domids, sizeof(domids));

        memset(page, 0, sizeof(page));

        new_maps = ioreq->v.niov;

    }



    if (batch_maps && new_maps) {

        ioreq->pages = xc_gnttab_map_grant_refs

            (gnt, new_maps, domids, refs, ioreq->prot);

        if (ioreq->pages == NULL) {

            xen_be_printf(&ioreq->blkdev->xendev, 0,

                          "can't map %d grant refs (%s, %d maps)\n",

                          new_maps, strerror(errno), ioreq->blkdev->cnt_map);

            return -1;

        }

        for (i = 0, j = 0; i < ioreq->v.niov; i++) {

            if (page[i] == NULL) {

                page[i] = ioreq->pages + (j++) * XC_PAGE_SIZE;

            }

        }

        ioreq->blkdev->cnt_map += new_maps;

    } else if (new_maps)  {

        for (i = 0; i < new_maps; i++) {

            ioreq->page[i] = xc_gnttab_map_grant_ref

                (gnt, domids[i], refs[i], ioreq->prot);

            if (ioreq->page[i] == NULL) {

                xen_be_printf(&ioreq->blkdev->xendev, 0,

                              "can't map grant ref %d (%s, %d maps)\n",

                              refs[i], strerror(errno), ioreq->blkdev->cnt_map);

                ioreq->mapped = 1;

                ioreq_unmap(ioreq);

                return -1;

            }

            ioreq->blkdev->cnt_map++;

        }

        for (i = 0, j = 0; i < ioreq->v.niov; i++) {

            if (page[i] == NULL) {

                page[i] = ioreq->page[j++];

            }

        }

    }

    if (ioreq->blkdev->feature_persistent) {

        while ((ioreq->blkdev->persistent_gnt_count < ioreq->blkdev->max_grants)

              && new_maps) {

            /* Go through the list of newly mapped grants and add as many

             * as possible to the list of persistently mapped grants.

             *

             * Since we start at the end of ioreq->page(s), we only need

             * to decrease new_maps to prevent this granted pages from

             * being unmapped in ioreq_unmap.

             */

            grant = g_malloc0(sizeof(*grant));

            new_maps--;

            if (batch_maps) {

                grant->page = ioreq->pages + (new_maps) * XC_PAGE_SIZE;

            } else {

                grant->page = ioreq->page[new_maps];

            }

            grant->blkdev = ioreq->blkdev;

            xen_be_printf(&ioreq->blkdev->xendev, 3,

                          "adding grant %" PRIu32 " page: %p\n",

                          refs[new_maps], grant->page);

            g_tree_insert(ioreq->blkdev->persistent_gnts,

                          GUINT_TO_POINTER(refs[new_maps]),

                          grant);

            ioreq->blkdev->persistent_gnt_count++;

        }

    }

    for (i = 0; i < ioreq->v.niov; i++) {

        ioreq->v.iov[i].iov_base += (uintptr_t)page[i];

    }

    ioreq->mapped = 1;

    ioreq->num_unmap = new_maps;

    return 0;

}
