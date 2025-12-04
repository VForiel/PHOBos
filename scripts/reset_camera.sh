#!/bin/bash

# Script to reset Point Grey USB camera using lsusb matching
# Finds the device by searching for the string "Point Grey" in lsusb output,
# then maps Bus/Device to the corresponding sysfs USB ID (e.g., 1-3) and performs
# an unbind/bind to reset the USB connection.

set -euo pipefail

SEARCH_TERM="Point Grey"

echo "Searching for USB device matching: $SEARCH_TERM (via lsusb)"

LSUSB_LINE=$(lsusb | grep -i "$SEARCH_TERM" | head -n 1 || true)

if [ -z "$LSUSB_LINE" ]; then
    echo "Error: No device found in lsusb matching '$SEARCH_TERM'."
    echo "Hint: Try 'lsusb' manually to confirm the camera is enumerated."
    exit 1
fi

# Example lsusb line:
# Bus 001 Device 004: ID 1e10:2000 Point Grey Research, Inc. Camera
BUS_NUM=$(echo "$LSUSB_LINE" | awk '{print $2}' | sed 's/^0*//')
DEV_NUM=$(echo "$LSUSB_LINE" | awk '{print $4}' | sed 's/://; s/^0*//')

echo "Found lsusb entry: Bus=$BUS_NUM Device=$DEV_NUM"

# Map to sysfs USB device by matching busnum/devnum files
USB_ID=""
for DEVPATH in /sys/bus/usb/devices/*; do
    # Skip non-directories
    [ -d "$DEVPATH" ] || continue
    # Some entries are interfaces (like 1-3:1.0); skip those
    if [[ "$(basename "$DEVPATH")" == *:* ]]; then
        continue
    fi
    if [ -r "$DEVPATH/busnum" ] && [ -r "$DEVPATH/devnum" ]; then
        BN=$(cat "$DEVPATH/busnum" 2>/dev/null || echo "")
        DN=$(cat "$DEVPATH/devnum" 2>/dev/null || echo "")
        # Normalize potential leading zeros from sysfs values as well
        BN_NORM=$(echo "$BN" | sed 's/^0*//')
        DN_NORM=$(echo "$DN" | sed 's/^0*//')
        if [ "$BN_NORM" = "$BUS_NUM" ] && [ "$DN_NORM" = "$DEV_NUM" ]; then
            USB_ID=$(basename "$DEVPATH")
            break
        fi
    fi
done

if [ -z "$USB_ID" ]; then
    echo "Error: Could not map Bus/Device to sysfs USB ID."
    echo "This can happen if permissions are restricted or the device changed state."
    exit 1
fi

echo "Resolved sysfs USB ID: $USB_ID"
echo "Attempting reset (requires sudo)…"

echo "Unbinding…"
if ! echo -n "$USB_ID" | sudo tee /sys/bus/usb/drivers/usb/unbind > /dev/null; then
    echo "Error: Failed to unbind. Make sure you have sudo privileges."
    exit 1
fi

sleep 1

echo "Rebinding…"
if ! echo -n "$USB_ID" | sudo tee /sys/bus/usb/drivers/usb/bind > /dev/null; then
    echo "Error: Failed to rebind."
    exit 1
fi

echo "Success: Point Grey camera reset complete."
