int pci_parse_devaddr(const char *addr, int *domp, int *busp,

                      unsigned int *slotp, unsigned int *funcp)

{

    const char *p;

    char *e;

    unsigned long val;

    unsigned long dom = 0, bus = 0;

    unsigned int slot = 0;

    unsigned int func = 0;



    p = addr;

    val = strtoul(p, &e, 16);

    if (e == p)

	return -1;

    if (*e == ':') {

	bus = val;

	p = e + 1;

	val = strtoul(p, &e, 16);

	if (e == p)

	    return -1;

	if (*e == ':') {

	    dom = bus;

	    bus = val;

	    p = e + 1;

	    val = strtoul(p, &e, 16);

	    if (e == p)

		return -1;

	}

    }



    slot = val;



    if (funcp != NULL) {

        if (*e != '.')

            return -1;



        p = e + 1;

        val = strtoul(p, &e, 16);

        if (e == p)

            return -1;



        func = val;

    }



    /* if funcp == NULL func is 0 */

    if (dom > 0xffff || bus > 0xff || slot > 0x1f || func > 7)

	return -1;



    if (*e)

	return -1;



    /* Note: QEMU doesn't implement domains other than 0 */

    if (!pci_find_bus(pci_find_root_bus(dom), bus))

	return -1;



    *domp = dom;

    *busp = bus;

    *slotp = slot;

    if (funcp != NULL)

        *funcp = func;

    return 0;

}
