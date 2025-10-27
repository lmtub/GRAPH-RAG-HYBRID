static void i440fx_update_memory_mappings(PCII440FXState *d)

{

    int i, r;

    uint32_t smram;

    bool smram_enabled;



    memory_region_transaction_begin();

    update_pam(d, 0xf0000, 0x100000, (d->dev.config[I440FX_PAM] >> 4) & 3,

               &d->pam_regions[0]);

    for(i = 0; i < 12; i++) {

        r = (d->dev.config[(i >> 1) + (I440FX_PAM + 1)] >> ((i & 1) * 4)) & 3;

        update_pam(d, 0xc0000 + 0x4000 * i, 0xc0000 + 0x4000 * (i + 1), r,

                   &d->pam_regions[i+1]);

    }

    smram = d->dev.config[I440FX_SMRAM];

    smram_enabled = (d->smm_enabled && (smram & 0x08)) || (smram & 0x40);

    memory_region_set_enabled(&d->smram_region, !smram_enabled);

    memory_region_transaction_commit();

}
