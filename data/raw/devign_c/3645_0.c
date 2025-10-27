void watchdog_add_model(WatchdogTimerModel *model)

{

    LIST_INSERT_HEAD(&watchdog_list, model, entry);

}
