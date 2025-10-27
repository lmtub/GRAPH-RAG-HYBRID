e1000e_set_pbaclr(E1000ECore *core, int index, uint32_t val)

{

    int i;



    core->mac[PBACLR] = val & E1000_PBACLR_VALID_MASK;



    if (msix_enabled(core->owner)) {

        return;

    }



    for (i = 0; i < E1000E_MSIX_VEC_NUM; i++) {

        if (core->mac[PBACLR] & BIT(i)) {

            msix_clr_pending(core->owner, i);

        }

    }

}
