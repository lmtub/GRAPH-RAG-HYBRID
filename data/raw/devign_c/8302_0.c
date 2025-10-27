e1000e_intrmgr_delay_tx_causes(E1000ECore *core, uint32_t *causes)

{

    static const uint32_t delayable_causes = E1000_ICR_TXQ0 |

                                             E1000_ICR_TXQ1 |

                                             E1000_ICR_TXQE |

                                             E1000_ICR_TXDW;



    if (msix_enabled(core->owner)) {

        return false;

    }



    /* Clean up all causes that may be delayed */

    core->delayed_causes |= *causes & delayable_causes;

    *causes &= ~delayable_causes;



    /* If there are causes that cannot be delayed */

    if (causes != 0) {

        return false;

    }



    /* All causes delayed */

    e1000e_intrmgr_rearm_timer(&core->tidv);



    if (!core->tadv.running && (core->mac[TADV] != 0)) {

        e1000e_intrmgr_rearm_timer(&core->tadv);

    }



    return true;

}
