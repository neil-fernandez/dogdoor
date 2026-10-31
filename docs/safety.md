# Safety Requirements & FMEA

A dog door is a **safety-critical device**: it moves a panel that could pinch a tail, leg, paw, or a
child's hand, and it gates an animal's access to shelter. Safety is designed in, not added after.

## 1. Hard safety requirements

1. **No pinch / no crush.** The flap must stop and reverse on any obstruction before it can cause
   harm. This must be enforced by *multiple independent means*, never a single sensor.
2. **Low maximum force.** "Soft close" means limited force — by construction it must be unable to
   injure even on worst case.
3. **Animal fully cleared before closing.** The door never closes while an animal (or anything) is
   in the doorway.
4. **Defined fail-safe.** Power loss and software faults resolve to a known, safe state.
5. **Manual override.** A human can always open the door by hand / kill power.
6. **No unauthorised entry.** Enrolled animals only; the door locks against anything else.
7. **Deterministic safety.** All safety logic runs on the microcontroller and does not depend on ML.

## 2. FMEA (Failure Modes and Effects Analysis)

Severity (S): 1 = negligible, 2 = minor, 3 = moderate, 4 = major, 5 = critical.
Likelihood (L): 1 = rare, 2 = unlikely, 3 = possible, 4 = likely, 5 = frequent.
Risk = S × L (acts on anything ≥ 8; must mitigate ≥ 12).

| # | Failure mode | S | L | Risk | Effect | Mitigation |
|---|---|---|---|---|---|---|
| 1 | Flap pinches tail/leg/paw during close | 5 | 3 | 15 | Injury | Stall detection (TMC2209) + IR break-beam + low force limit + instant reverse |
| 2 | Flap closes while animal partway through | 5 | 3 | 15 | Injury / trap | "Cleared" logic (dual beam + hold delay) + vision "fully through" check |
| 3 | Unauthorised animal (cat/raccoon) enters | 3 | 3 | 9 | Unwanted entry | RFID gate + vision reject "other" + locking |
| 4 | Animal locked out on power loss | 4 | 2 | 8 | Stranded animal | Battery backup + fail-open-or-manual decision + manual release |
| 5 | Motor stall / overheat | 3 | 3 | 9 | Damage / fire risk | Current limit, timeout, thermal cutout, driver stall shutdown |
| 6 | Child's hand caught | 5 | 2 | 10 | Injury | Low force + beam + reverse (same as #1) + high mount/guards |
| 7 | Water ingress → short circuit | 4 | 3 | 12 | Failure / shock | Drip-lip, perimeter seal, conformal coating, sealed enclosure |
| 8 | Bug ingress when closed | 2 | 4 | 8 | Hygiene | Magnetic catch + full-close + perimeter brush/EPDM seal |
| 9 | Vision mis-identifies animal | 3 | 4 | 12 | Wrong open/close | RFID authoritative; vision advisory only; threshold + "other" gate |
| 10 | RFID reader fails → dog locked out | 4 | 2 | 8 | Stranded animal | Vision fallback + manual override |
| 11 | Hinge/motor mechanical failure | 3 | 2 | 6 | Door stuck | Manual override, serviceable design |
| 12 | Software hang (Pi or ESP32) | 4 | 2 | 8 | Uncontrolled state | ESP32 watchdog timer → defined fail-safe state |

## 3. Sensor redundancy matrix (the "never one sensor" rule)

| Hazard | Primary | Independent backup |
|--------|---------|--------------------|
| Obstruction in path | IR break-beam | Stepper stall detection |
| Force too high | Current/stall limit | Mechanical force limit (slip clutch) |
| Door position wrong | Limit switches | Encoder/step count cross-check |
| Animal present | RFID | Vision (and vice-versa) |
| System hung | Watchdog timer | Fail-safe hardware default |

## 4. Decisions to lock in Phase 0

- **Fail-safe on power loss:** fail-open (animal safety) vs fail-closed (security) — decide
  explicitly. Default recommendation: fail-**closed** with a **manual mechanical release** + battery
  backup, because an external door left open is a security/weather risk.
- **Maximum closing force:** measure and specify (e.g. must reverse above X N).
- **"Animal cleared" definition:** exact beam layout + hold time.
