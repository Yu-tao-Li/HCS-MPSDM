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

BuildSAC3 $massScale $stiffnessScale $strengthScale $ductilityScale $story2K $story3K
set lambda [eigen -fullGenLapack 3]
set outputDir [file join $here outputs]
file mkdir $outputDir
modalProperties -file [file join $outputDir modal_properties.txt] -unorm
set pi [expr acos(-1.0)]

set out [open [file join $outputDir modal_results.csv] w]
puts $out "mode,eigenvalue,period_s"
set mode 0
foreach lam $lambda {
    incr mode
    set T [expr 2.0*$pi/sqrt($lam)]
    puts $out "$mode,$lam,$T"
    puts "MODE $mode PERIOD $T"
}
close $out
