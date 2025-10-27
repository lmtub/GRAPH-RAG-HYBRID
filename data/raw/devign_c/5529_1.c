static int pci_rocker_init(PCIDevice *dev)

{

    Rocker *r = to_rocker(dev);

    const MACAddr zero = { .a = { 0, 0, 0, 0, 0, 0 } };

    const MACAddr dflt = { .a = { 0x52, 0x54, 0x00, 0x12, 0x35, 0x01 } };

    static int sw_index;

    int i, err = 0;



    /* allocate worlds */



    r->worlds[ROCKER_WORLD_TYPE_OF_DPA] = of_dpa_world_alloc(r);



    if (!r->world_name) {

        r->world_name = g_strdup(world_name(r->worlds[ROCKER_WORLD_TYPE_OF_DPA]));

    }



    r->world_dflt = rocker_world_type_by_name(r, r->world_name);

    if (!r->world_dflt) {

        fprintf(stderr,

                "rocker: requested world \"%s\" does not exist\n",

                r->world_name);

        err = -EINVAL;

        goto err_world_type_by_name;

    }



    /* set up memory-mapped region at BAR0 */



    memory_region_init_io(&r->mmio, OBJECT(r), &rocker_mmio_ops, r,

                          "rocker-mmio", ROCKER_PCI_BAR0_SIZE);

    pci_register_bar(dev, ROCKER_PCI_BAR0_IDX,

                     PCI_BASE_ADDRESS_SPACE_MEMORY, &r->mmio);



    /* set up memory-mapped region for MSI-X */



    memory_region_init(&r->msix_bar, OBJECT(r), "rocker-msix-bar",

                       ROCKER_PCI_MSIX_BAR_SIZE);

    pci_register_bar(dev, ROCKER_PCI_MSIX_BAR_IDX,

                     PCI_BASE_ADDRESS_SPACE_MEMORY, &r->msix_bar);



    /* MSI-X init */



    err = rocker_msix_init(r);

    if (err) {

        goto err_msix_init;

    }



    /* validate switch properties */



    if (!r->name) {

        r->name = g_strdup(ROCKER);

    }



    if (rocker_find(r->name)) {

        err = -EEXIST;

        goto err_duplicate;

    }



    /* Rocker name is passed in port name requests to OS with the intention

     * that the name is used in interface names. Limit the length of the

     * rocker name to avoid naming problems in the OS. Also, adding the

     * port number as p# and unganged breakout b#, where # is at most 2

     * digits, so leave room for it too (-1 for string terminator, -3 for

     * p# and -3 for b#)

     */

#define ROCKER_IFNAMSIZ 16

#define MAX_ROCKER_NAME_LEN  (ROCKER_IFNAMSIZ - 1 - 3 - 3)

    if (strlen(r->name) > MAX_ROCKER_NAME_LEN) {

        fprintf(stderr,

                "rocker: name too long; please shorten to at most %d chars\n",

                MAX_ROCKER_NAME_LEN);

        return -EINVAL;

    }



    if (memcmp(&r->fp_start_macaddr, &zero, sizeof(zero)) == 0) {

        memcpy(&r->fp_start_macaddr, &dflt, sizeof(dflt));

        r->fp_start_macaddr.a[4] += (sw_index++);

    }



    if (!r->switch_id) {

        memcpy(&r->switch_id, &r->fp_start_macaddr,

               sizeof(r->fp_start_macaddr));

    }



    if (r->fp_ports > ROCKER_FP_PORTS_MAX) {

        r->fp_ports = ROCKER_FP_PORTS_MAX;

    }



    r->rings = g_new(DescRing *, rocker_pci_ring_count(r));



    /* Rings are ordered like this:

     * - command ring

     * - event ring

     * - port0 tx ring

     * - port0 rx ring

     * - port1 tx ring

     * - port1 rx ring

     * .....

     */



    for (i = 0; i < rocker_pci_ring_count(r); i++) {

        DescRing *ring = desc_ring_alloc(r, i);



        if (i == ROCKER_RING_CMD) {

            desc_ring_set_consume(ring, cmd_consume, ROCKER_MSIX_VEC_CMD);

        } else if (i == ROCKER_RING_EVENT) {

            desc_ring_set_consume(ring, NULL, ROCKER_MSIX_VEC_EVENT);

        } else if (i % 2 == 0) {

            desc_ring_set_consume(ring, tx_consume,

                                  ROCKER_MSIX_VEC_TX((i - 2) / 2));

        } else if (i % 2 == 1) {

            desc_ring_set_consume(ring, NULL, ROCKER_MSIX_VEC_RX((i - 3) / 2));

        }



        r->rings[i] = ring;

    }



    for (i = 0; i < r->fp_ports; i++) {

        FpPort *port =

            fp_port_alloc(r, r->name, &r->fp_start_macaddr,

                          i, &r->fp_ports_peers[i]);



        r->fp_port[i] = port;

        fp_port_set_world(port, r->world_dflt);

    }



    QLIST_INSERT_HEAD(&rockers, r, next);



    return 0;



err_duplicate:

    rocker_msix_uninit(r);

err_msix_init:

    object_unparent(OBJECT(&r->msix_bar));

    object_unparent(OBJECT(&r->mmio));

err_world_type_by_name:

    for (i = 0; i < ROCKER_WORLD_TYPE_MAX; i++) {

        if (r->worlds[i]) {

            world_free(r->worlds[i]);

        }

    }

    return err;

}
