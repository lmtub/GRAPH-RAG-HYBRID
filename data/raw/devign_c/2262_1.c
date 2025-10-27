static uint32_t timer_int_route(struct HPETTimer *timer)

{

    uint32_t route;

    route = (timer->config & HPET_TN_INT_ROUTE_MASK) >> HPET_TN_INT_ROUTE_SHIFT;

    return route;

}
