void omap_clk_init(struct omap_mpu_state_s *mpu)

{

    struct clk **i, *j, *k;

    int count;

    int flag;



    if (cpu_is_omap310(mpu))

        flag = CLOCK_IN_OMAP310;

    else if (cpu_is_omap1510(mpu))

        flag = CLOCK_IN_OMAP1510;

    else if (cpu_is_omap2410(mpu) || cpu_is_omap2420(mpu))

        flag = CLOCK_IN_OMAP242X;

    else if (cpu_is_omap2430(mpu))

        flag = CLOCK_IN_OMAP243X;

    else if (cpu_is_omap3430(mpu))

        flag = CLOCK_IN_OMAP243X;

    else

        return;



    for (i = onchip_clks, count = 0; *i; i ++)

        if ((*i)->flags & flag)

            count ++;

    mpu->clks = (struct clk *) g_malloc0(sizeof(struct clk) * (count + 1));

    for (i = onchip_clks, j = mpu->clks; *i; i ++)

        if ((*i)->flags & flag) {

            memcpy(j, *i, sizeof(struct clk));

            for (k = mpu->clks; k < j; k ++)

                if (j->parent && !strcmp(j->parent->name, k->name)) {

                    j->parent = k;

                    j->sibling = k->child1;

                    k->child1 = j;

                } else if (k->parent && !strcmp(k->parent->name, j->name)) {

                    k->parent = j;

                    k->sibling = j->child1;

                    j->child1 = k;

                }

            j->divisor = j->divisor ?: 1;

            j->multiplier = j->multiplier ?: 1;

            j ++;

        }

    for (j = mpu->clks; count --; j ++) {

        omap_clk_update(j);

        omap_clk_rate_update(j);

    }

}
