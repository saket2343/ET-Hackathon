---
doc_no: SOP-315
title: Vibration Monitoring and Alarm Response — Area 1 Pumps
revision: 2
effective_date: 2025-06-30
governs: [P-101, P-102, VT-101]
type: SOP
---

# SOP-315 — Vibration Monitoring and Alarm Response

## 1. Monitoring
Pump P-101 is continuously monitored by vibration transmitter VT-101 mounted on
the drive-end bearing housing. Readings are velocity RMS in mm/s.

## 2. Alarm Limits (per ISO 10816-3, Group 2 machines)
- Zone A/B boundary (good):    2.8 mm/s
- Zone B/C boundary (alert):   4.5 mm/s — investigate within 48 hours
- Zone C/D boundary (danger):  7.1 mm/s — shut down and isolate immediately

## 3. Alert Response (4.5–7.1 mm/s)
1. Confirm the reading with a portable meter at the VT-101 location.
2. Check bearing housing temperature; above 85 °C indicates lubrication failure.
3. Review the trend: a rising trend over more than 24 hours with a 1x running
   speed signature indicates bearing wear or misalignment.
4. Raise a work order and schedule bearing inspection per SOP-207 before the
   trend reaches the 7.1 mm/s danger limit.

## 4. Danger Response (above 7.1 mm/s)
Stop the pump immediately from HS-101 and isolate per SAF-12. Running in Zone D
risks catastrophic bearing seizure and shaft damage.
