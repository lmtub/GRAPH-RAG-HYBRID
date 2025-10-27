static void omap_pin_cfg_write(void *opaque, target_phys_addr_t addr,

                               uint64_t value, unsigned size)

{

    struct omap_mpu_state_s *s = (struct omap_mpu_state_s *) opaque;

    uint32_t diff;



    if (size != 4) {

        return omap_badwidth_write32(opaque, addr, value);

    }



    switch (addr) {

    case 0x00:	/* FUNC_MUX_CTRL_0 */

        diff = s->func_mux_ctrl[addr >> 2] ^ value;

        s->func_mux_ctrl[addr >> 2] = value;

        omap_pin_funcmux0_update(s, diff, value);

        return;



    case 0x04:	/* FUNC_MUX_CTRL_1 */

        diff = s->func_mux_ctrl[addr >> 2] ^ value;

        s->func_mux_ctrl[addr >> 2] = value;

        omap_pin_funcmux1_update(s, diff, value);

        return;



    case 0x08:	/* FUNC_MUX_CTRL_2 */

        s->func_mux_ctrl[addr >> 2] = value;

        return;



    case 0x0c:	/* COMP_MODE_CTRL_0 */

        s->comp_mode_ctrl[0] = value;

        s->compat1509 = (value != 0x0000eaef);

        omap_pin_funcmux0_update(s, ~0, s->func_mux_ctrl[0]);

        omap_pin_funcmux1_update(s, ~0, s->func_mux_ctrl[1]);

        return;



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

        s->func_mux_ctrl[(addr >> 2) - 1] = value;

        return;



    case 0x40:	/* PULL_DWN_CTRL_0 */

    case 0x44:	/* PULL_DWN_CTRL_1 */

    case 0x48:	/* PULL_DWN_CTRL_2 */

    case 0x4c:	/* PULL_DWN_CTRL_3 */

        s->pull_dwn_ctrl[(addr & 0xf) >> 2] = value;

        return;



    case 0x50:	/* GATE_INH_CTRL_0 */

        s->gate_inh_ctrl[0] = value;

        return;



    case 0x60:	/* VOLTAGE_CTRL_0 */

        s->voltage_ctrl[0] = value;

        return;



    case 0x70:	/* TEST_DBG_CTRL_0 */

        s->test_dbg_ctrl[0] = value;

        return;



    case 0x80:	/* MOD_CONF_CTRL_0 */

        diff = s->mod_conf_ctrl[0] ^ value;

        s->mod_conf_ctrl[0] = value;

        omap_pin_modconf1_update(s, diff, value);

        return;



    default:

        OMAP_BAD_REG(addr);

    }

}
