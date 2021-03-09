#!/usr/bin/perl
use strict;
use warnings;

my $file = $ARGV[0] or die "Need to get CSV file on the command line\n";

#my $field1, $field2, $field3, $field4;
open(my $data, '<', $file) or die "Could not open '$file' $!\n";
open(my $out_file, '>', 'testvec_for_fpga.txt');

while (my $line = <$data>) {
  chomp $line;

  my @fields = split "," , $line;
  my $word128 = $fields[3];
  my $address = $fields[2];
  my $word64_1 = substr($word128, 0, 18);
  my $word64_0 = substr($word128, -16);

  #print "$word64_0 \n";
  print $out_file "   weight128.val64[1] = $word64_1; weight128.val64[0] = 0x$word64_0; HW128_REG($address) = weight128.val128;\n";
  #print $out_file "$address,$word64_1,0x$word64_0\\n";
  #print "$address,$word64_1,0x$word64_0\\n";
}

close $data;
close $out_file;
