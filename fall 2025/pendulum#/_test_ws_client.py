import asyncio
import websockets
import json

async def test():
    uri = 'ws://127.0.0.1:8000/ws/train?method=q&episodes=1'
    try:
        async with websockets.connect(uri) as ws:
            print('connected')
            while True:
                msg = await ws.recv()
                data = json.loads(msg)
                print('recv', data.get('type'), 'ep', data.get('episode'), 'step', data.get('step'))
                if data.get('type') == 'complete':
                    break
    except Exception as e:
        print('client error', e)

if __name__ == '__main__':
    asyncio.run(test())
