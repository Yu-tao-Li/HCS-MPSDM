set here [file dirname [info script]]
source [file join $here model.tcl]

set massScale 1.0
set stiffnessScale 1.071
set strengthScale 0.76
set ductilityScale 0.42
set story2K 1.0
set story3K 1.0
if {$argc >= 1} {set massScale [lindex $argv 0]}
if {$argc >= 2} {set stiffnessScale [lindex $argv 1]}
if {$argc >= 3} {set strengthScale [lindex $argv 2]}
if {$argc >= 4} {set ductilityScale [lindex $argv 3]}
if {$argc >= 5} {set story2K [lindex $argv 4]}
if {$argc >= 6} {set story3K [lindex $argv 5]}

set meta [BuildSAC3 $massScale $stiffnessScale $strengthScale $ductilityScale $story2K $story3K]
set H [lindex $meta 0]
set m1 [lindex $meta 3]
set m2 [lindex $meta 4]
set m3 [lindex $meta 5]
set totalWeight [expr ($m1+$m2+$m3)*9810.0]

# First-mode-like lateral load profile.
timeSeries Linear 2
pattern Plain 2 2 {
    load 11 [expr $m1*1.0] 0.0 0.0
    load 21 [expr $m2*2.0] 0.0 0.0
    load 31 [expr $m3*3.0] 0.0 0.0
}

wipeAnalysis
constraints Transformation
numberer RCM
system UmfPack
test NormDispIncr 1.0e-6 100 0
algorithm Newton
set dU 0.5
integrator DisplacementControl 31 1 $dU
analysis Static

set outputDir [file join $here outputs]
file mkdir $outputDir
set out [open [file join $outputDir pushover_results.csv] w]
puts $out "roof_drift,normalized_base_shear,roof_displacement_mm,load_factor"

set targetDisp [expr 0.05*3.0*$H]
set ok 0
while {[nodeDisp 31 1] < $targetDisp && $ok == 0} {
    set ok [analyze 1]
    if {$ok != 0} {
        test NormDispIncr 1.0e-5 300 0
        algorithm NewtonLineSearch -type Bisection
        set ok [analyze 1]
    }
    if {$ok != 0} {
        algorithm KrylovNewton
        set ok [analyze 1]
    }
    if {$ok != 0} {
        algorithm Broyden 20
        set ok [analyze 1]
    }
    if {$ok != 0} {
        algorithm ModifiedNewton -initial
        set ok [analyze 1]
    }
    if {$ok == 0} {
        algorithm Newton
        test NormDispIncr 1.0e-6 100 0
    }
    if {$ok == 0} {
        reactions
        set baseShear 0.0
        foreach nodeTag {1 2 3 4 9000} {
            set baseShear [expr $baseShear-[nodeReaction $nodeTag 1]]
        }
        set roofDisp [nodeDisp 31 1]
        set drift [expr $roofDisp/(3.0*$H)]
        set normV [expr $baseShear/$totalWeight]
        puts $out "$drift,$normV,$roofDisp,[getTime]"
    }
}
close $out
puts "PUSHOVER_STATUS $ok"
