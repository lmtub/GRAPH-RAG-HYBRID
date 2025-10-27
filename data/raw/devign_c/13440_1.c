e1000e_init_msix(E1000EState *s)

{

    PCIDevice *d = PCI_DEVICE(s);

    int res = msix_init(PCI_DEVICE(s), E1000E_MSIX_VEC_NUM,

                        &s->msix,

                        E1000E_MSIX_IDX, E1000E_MSIX_TABLE,

                        &s->msix,

                        E1000E_MSIX_IDX, E1000E_MSIX_PBA,

                        0xA0);



    if (res < 0) {

        trace_e1000e_msix_init_fail(res);

    } else {

        if (!e1000e_use_msix_vectors(s, E1000E_MSIX_VEC_NUM)) {

            msix_uninit(d, &s->msix, &s->msix);

        }

    }

}
