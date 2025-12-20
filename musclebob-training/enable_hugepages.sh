#!/bin/bash
# Script to enable transparent hugepages for TPU performance
# This addresses the warning about transparent hugepages not being enabled

echo "Checking current transparent hugepage status..."
if [ -f /sys/kernel/mm/transparent_hugepage/enabled ]; then
    cat /sys/kernel/mm/transparent_hugepage/enabled
else
    echo "Transparent hugepage settings not found (not on a TPU VM?)"
    exit 0
fi

echo ""
echo "Attempting to enable transparent hugepages..."
sudo sh -c "echo always > /sys/kernel/mm/transparent_hugepage/enabled"

if [ $? -eq 0 ]; then
    echo "✓ Successfully enabled transparent hugepages"
    echo ""
    echo "New setting:"
    cat /sys/kernel/mm/transparent_hugepage/enabled
else
    echo "✗ Failed to enable transparent hugepages"
    echo "You may need to run this script with sudo privileges"
fi
