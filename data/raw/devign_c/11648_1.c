uint32_t hpet_in_legacy_mode(void)

{

    if (hpet_statep)

        return hpet_statep->config & HPET_CFG_LEGACY;

    else

        return 0;

}
