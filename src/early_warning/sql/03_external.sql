DROP SCHEMA IF EXISTS __EXT__ CASCADE;
CREATE SCHEMA __EXT__;
CREATE TABLE __EXT__.holidays (
    holiday_date date PRIMARY KEY,
    name         text NOT NULL,
    source       text NOT NULL
);
