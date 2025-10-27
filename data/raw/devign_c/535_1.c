e1000e_cleanup_msix(E1000EState *s)

{

    if (msix_enabled(PCI_DEVICE(s))) {

        e1000e_unuse_msix_vectors(s, E1000E_MSIX_VEC_NUM);

        msix_uninit(PCI_DEVICE(s), &s->msix, &s->msix);

    }

}
