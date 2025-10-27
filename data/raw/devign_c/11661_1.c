static int xenfb_map_fb(struct XenFB *xenfb)

{

    struct xenfb_page *page = xenfb->c.page;

    char *protocol = xenfb->c.xendev.protocol;

    int n_fbdirs;

    xen_pfn_t *pgmfns = NULL;

    xen_pfn_t *fbmfns = NULL;

    void *map, *pd;

    int mode, ret = -1;



    /* default to native */

    pd = page->pd;

    mode = sizeof(unsigned long) * 8;



    if (!protocol) {

	/*

	 * Undefined protocol, some guesswork needed.

	 *

	 * Old frontends which don't set the protocol use

	 * one page directory only, thus pd[1] must be zero.

	 * pd[1] of the 32bit struct layout and the lower

	 * 32 bits of pd[0] of the 64bit struct layout have

	 * the same location, so we can check that ...

	 */

	uint32_t *ptr32 = NULL;

	uint32_t *ptr64 = NULL;

#if defined(__i386__)

	ptr32 = (void*)page->pd;

	ptr64 = ((void*)page->pd) + 4;

#elif defined(__x86_64__)

	ptr32 = ((void*)page->pd) - 4;

	ptr64 = (void*)page->pd;

#endif

	if (ptr32) {

	    if (ptr32[1] == 0) {

		mode = 32;

		pd   = ptr32;

	    } else {

		mode = 64;

		pd   = ptr64;

	    }

	}

#if defined(__x86_64__)

    } else if (strcmp(protocol, XEN_IO_PROTO_ABI_X86_32) == 0) {

	/* 64bit dom0, 32bit domU */

	mode = 32;

	pd   = ((void*)page->pd) - 4;

#elif defined(__i386__)

    } else if (strcmp(protocol, XEN_IO_PROTO_ABI_X86_64) == 0) {

	/* 32bit dom0, 64bit domU */

	mode = 64;

	pd   = ((void*)page->pd) + 4;

#endif

    }



    if (xenfb->pixels) {

        munmap(xenfb->pixels, xenfb->fbpages * XC_PAGE_SIZE);

        xenfb->pixels = NULL;

    }



    xenfb->fbpages = (xenfb->fb_len + (XC_PAGE_SIZE - 1)) / XC_PAGE_SIZE;

    n_fbdirs = xenfb->fbpages * mode / 8;

    n_fbdirs = (n_fbdirs + (XC_PAGE_SIZE - 1)) / XC_PAGE_SIZE;



    pgmfns = g_malloc0(sizeof(xen_pfn_t) * n_fbdirs);

    fbmfns = g_malloc0(sizeof(xen_pfn_t) * xenfb->fbpages);



    xenfb_copy_mfns(mode, n_fbdirs, pgmfns, pd);

    map = xc_map_foreign_pages(xen_xc, xenfb->c.xendev.dom,

			       PROT_READ, pgmfns, n_fbdirs);

    if (map == NULL)

	goto out;

    xenfb_copy_mfns(mode, xenfb->fbpages, fbmfns, map);

    munmap(map, n_fbdirs * XC_PAGE_SIZE);



    xenfb->pixels = xc_map_foreign_pages(xen_xc, xenfb->c.xendev.dom,

            PROT_READ, fbmfns, xenfb->fbpages);

    if (xenfb->pixels == NULL)

	goto out;



    ret = 0; /* all is fine */



out:

    g_free(pgmfns);

    g_free(fbmfns);

    return ret;

}
