static inline void scoop_gpio_handler_update(ScoopInfo *s) {

    uint32_t level, diff;

    int bit;

    level = s->gpio_level & s->gpio_dir;



    for (diff = s->prev_level ^ level; diff; diff ^= 1 << bit) {

        bit = ffs(diff) - 1;

        qemu_set_irq(s->handler[bit], (level >> bit) & 1);

    }



    s->prev_level = level;

}
