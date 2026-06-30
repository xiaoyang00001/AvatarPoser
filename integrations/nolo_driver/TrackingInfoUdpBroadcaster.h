#pragma once

#include "packet_types.h"

#include <cstddef>
#include <cstdint>
#include <string>

class TrackingInfoUdpBroadcaster
{
public:
	static TrackingInfoUdpBroadcaster& Instance();

	void Broadcast(const TrackingInfo& info);

private:
	TrackingInfoUdpBroadcaster();
	~TrackingInfoUdpBroadcaster();

	TrackingInfoUdpBroadcaster(const TrackingInfoUdpBroadcaster&) = delete;
	TrackingInfoUdpBroadcaster& operator=(const TrackingInfoUdpBroadcaster&) = delete;

	bool Initialize();
	bool IsTrackingInfoValid(const TrackingInfo& info) const;
	void BuildPayload(char* payload, size_t payloadSize, int& payloadLength, const TrackingInfo& info) const;

	bool m_initialized;
	bool m_enabled;
	std::string m_host;
	int m_port;
	uintptr_t m_socket;
	uint32_t m_destinationAddress;
	uint16_t m_destinationPort;
	uint64_t m_lastBroadcastFrame;
	uint64_t m_invalidFrameCount;
};
