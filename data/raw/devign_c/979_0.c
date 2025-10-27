static uint64_t omap_pin_cfg_read(void *opaque, target_phys_addr_t addr,

                                  unsigned size)

{

    struct omap_mpu_state_s *s = (struct omap_mpu_state_s *) opaque;



    if (size != 4) {

        return omap_badwidth_read32(opaque, addr);

    }



    switch (addr) {

    case 0x00:	/* FUNC_MUX_CTRL_0 */

    case 0x04:	/* FUNC_MUX_CTRL_1 */

    case 0x08:	/* FUNC_MUX_CTRL_2 */

        return s->func_mux_ctrl[addr >> 2];



    case 0x0c:	/* COMP_MODE_CTRL_0 */

        return s->comp_mode_ctrl[0];



    case 0x10:	/* FUNC_MUX_CTRL_3 */

    case 0x14:	/* FUNC_MUX_CTRL_4 */

    case 0x18:	/* FUNC_MUX_CTRL_5 */

    case 0x1c:	/* FUNC_MUX_CTRL_6 */

    case 0x20:	/* FUNC_MUX_CTRL_7 */

    case 0x24:	/* FUNC_MUX_CTRL_8 */

    case 0x28:	/* FUNC_MUX_CTRL_9 */

    case 0x2c:	/* FUNC_MUX_CTRL_A */

    case 0x30:	/* FUNC_MUX_CTRL_B */

    case 0x34:	/* FUNC_MUX_CTRL_C */

    case 0x38:	/* FUNC_MUX_CTRL_D */

        return s->func_mux_ctrl[(addr >> 2) - 1];



    case 0x40:	/* PULL_DWN_CTRL_0 */

    case 0x44:	/* PULL_DWN_CTRL_1 */

    case 0x48:	/* PULL_DWN_CTRL_2 */

    case 0x4c:	/* PULL_DWN_CTRL_3 */

        return s->pull_dwn_ctrl[(addr & 0xf) >> 2];



    case 0x50:	/* GATE_INH_CTRL_0 */

        return s->gate_inh_ctrl[0];



    case 0x60:	/* VOLTAGE_CTRL_0 */

        return s->voltage_ctrl[0];



    case 0x70:	/* TEST_DBG_CTRL_0 */

        return s->test_dbg_ctrl[0];



    case 0x80:	/* MOD_CONF_CTRL_0 */

        return s->mod_conf_ctrl[0];

    }



    OMAP_BAD_REG(addr);

    return 0;

}
