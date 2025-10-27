static int check_bd(VIOsPAPRVLANDevice *dev, vlan_bd_t bd,

                    target_ulong alignment)

{

    if ((VLAN_BD_ADDR(bd) % alignment)

        || (VLAN_BD_LEN(bd) % alignment)) {

        return -1;

    }



    if (spapr_vio_check_tces(&dev->sdev, VLAN_BD_ADDR(bd),

                             VLAN_BD_LEN(bd), SPAPR_TCE_RW) != 0) {

        return -1;

    }



    return 0;

}
